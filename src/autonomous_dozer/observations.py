"""Partial height observations from an idealized gridded surface sensor.

Only this simulator sees Terrain truth. Downstream estimators receive Observation,
which contains measurements and validity/uncertainty, never hidden heights.
"""
from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray
from .domain import Grid, Terrain, positive


@dataclass(frozen=True)
class SensorPose:
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in self.__dict__.values()):
            raise ValueError("Sensor pose must be finite")


@dataclass(frozen=True)
class SensorModel:
    min_range_m: float = 0.25
    max_range_m: float = 25.0
    horizontal_fov_rad: float = math.tau
    height_noise_std_m: float = 0.02
    dropout_probability: float = 0.0
    max_ray_cell_visits: int = 20_000_000

    def __post_init__(self):
        positive("max_range_m", self.max_range_m)
        if not math.isfinite(self.min_range_m) or not 0 <= self.min_range_m < self.max_range_m:
            raise ValueError("Range limits require 0 <= min < max")
        if not math.isfinite(self.horizontal_fov_rad) or not 0 < self.horizontal_fov_rad <= math.tau:
            raise ValueError("Horizontal FOV must lie in (0, 2*pi]")
        if not math.isfinite(self.height_noise_std_m) or self.height_noise_std_m < 0:
            raise ValueError("Height-noise standard deviation must be finite and nonnegative")
        if not math.isfinite(self.dropout_probability) or not 0 <= self.dropout_probability <= 1:
            raise ValueError("Dropout probability must lie in [0, 1]")
        if type(self.max_ray_cell_visits) is not int or self.max_ray_cell_visits < 1:
            raise ValueError("Ray-cell budget must be a positive integer")


@dataclass(frozen=True)
class Observation:
    grid: Grid
    height_m: NDArray[np.float64]
    observed: NDArray[np.bool_]
    height_std_m: NDArray[np.float64]
    source_version: int
    stamp_s: float  # Simulated acquisition time, not wall time.
    sensor_pose: SensorPose
    sensor_model: SensorModel
    seed: int

    def __post_init__(self):
        h = np.array(self.height_m, dtype=float, copy=True)
        valid = np.array(self.observed, copy=True)
        std = np.array(self.height_std_m, dtype=float, copy=True)
        if h.shape != self.grid.shape or valid.shape != h.shape or std.shape != h.shape:
            raise ValueError("Observation arrays must match grid shape")
        if valid.dtype != np.bool_:
            raise ValueError("Observed mask must be boolean")
        if (not np.all(np.isfinite(h[valid])) or not np.all(np.isfinite(std[valid]))
                or np.any(std[valid] < 0)):
            raise ValueError("Observed cells need finite heights and nonnegative uncertainty")
        if not np.all(np.isnan(h[~valid])) or not np.all(np.isnan(std[~valid])):
            raise ValueError("Unobserved height/uncertainty must be NaN, never hidden truth or ground")
        if type(self.source_version) is not int or self.source_version < 0:
            raise ValueError("Source version must be a nonnegative integer")
        if not math.isfinite(self.stamp_s) or self.stamp_s < 0:
            raise ValueError("Acquisition time must be finite and nonnegative")
        _validate_seed(self.seed)
        for name, a in [("height_m", h), ("observed", valid), ("height_std_m", std)]:
            a.flags.writeable = False
            object.__setattr__(self, name, a)


def _validate_seed(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")


def _visible(terrain: Terrain, pose: SensorPose, row: int, col: int) -> bool:
    """Traverse ray/grid intersections exactly in XY (piecewise-flat columns).

    A cell blocks a ray when its top penetrates the ray before the target cell.
    Zero-length corner touches do not count as intersections. Ground is flat;
    there are no overhangs or outside-map obstacles in this representation.
    """
    g = terrain.grid
    u, v = (pose.x_m-g.origin_x_m)/g.resolution_m, (pose.y_m-g.origin_y_m)/g.resolution_m
    c, r = math.floor(u), math.floor(v)
    du, dv = col+0.5-u, row+0.5-v
    sx, sy = (1 if du > 0 else -1), (1 if dv > 0 else -1)
    dx = 1/abs(du) if du else math.inf
    dy = 1/abs(dv) if dv else math.inf
    tx = ((c+1-u) if du > 0 else (u-c))*dx if du else math.inf
    ty = ((r+1-v) if dv > 0 else (v-r))*dy if dv else math.inf
    entry = 0.0
    dz = float(terrain.height_m[row, col])-pose.z_m
    for _ in range(g.nx+g.ny+1):
        if r == row and c == col:
            # A top surface above the eye is hidden by its own vertical cell face.
            return min(pose.z_m+entry*dz, pose.z_m+dz) >= terrain.height_m[r, c]-1e-9
        end = min(tx, ty, 1.0)
        if end > entry:
            ray_bottom = min(pose.z_m+entry*dz, pose.z_m+end*dz)
            if terrain.height_m[r, c] > ray_bottom+1e-9:
                return False
        step_x, step_y = tx <= ty+1e-12, ty <= tx+1e-12
        if step_x:
            c += sx
            tx += dx
        if step_y:
            r += sy
            ty += dy
        entry = end
    raise RuntimeError("Ray traversal failed to reach its target")


def observe(terrain: Terrain, pose: SensorPose, model: SensorModel = SensorModel(), *,
            seed: int = 7, stamp_s: float = 0.0) -> Observation:
    """Sample visible cell tops, then apply dropout and unbiased vertical noise.

    Max/min range are 3D slant ranges. FOV is horizontal about sensor yaw. Noise
    is independent Gaussian height noise, not a calibrated LiDAR beam model.
    Negative noisy ground measurements are retained to avoid clipping bias.
    """
    _validate_seed(seed)
    if not math.isfinite(stamp_s) or stamp_s < 0:
        raise ValueError("Acquisition time must be finite and nonnegative")
    g, b = terrain.grid, terrain.grid.bounds
    if not (b.xmin_m <= pose.x_m < b.xmax_m and b.ymin_m <= pose.y_m < b.ymax_m):
        raise ValueError("Sensor XY must lie inside the map; outside geometry is unknown")
    c = math.floor((pose.x_m-g.origin_x_m)/g.resolution_m)
    r = math.floor((pose.y_m-g.origin_y_m)/g.resolution_m)
    if pose.z_m <= terrain.height_m[r, c]:
        raise ValueError("Sensor must be above the local terrain surface")
    x, y = g.centers()
    dx, dy = x-pose.x_m, y-pose.y_m
    distance = np.sqrt(dx**2+dy**2+(terrain.height_m-pose.z_m)**2)
    bearing = np.arctan2(dy, dx)-pose.yaw_rad
    bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
    candidates = ((distance >= model.min_range_m) & (distance <= model.max_range_m)
                  & ((np.abs(bearing) <= model.horizontal_fov_rad/2+1e-12)
                     | (np.hypot(dx, dy) <= 1e-12)))
    rows, cols = np.nonzero(candidates)
    upper_bound = int((np.abs(rows-r)+np.abs(cols-c)+1).sum())
    if upper_bound > model.max_ray_cell_visits:
        raise ValueError("Ray-cell budget exceeded; reduce map/range or explicitly increase budget")
    visible = np.zeros(g.shape, dtype=bool)
    for row, col in zip(rows, cols):
        visible[row, col] = _visible(terrain, pose, int(row), int(col))
    # Fixed-size independent streams keep noise unchanged when dropout changes.
    dropout_rng, noise_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    valid = visible & (dropout_rng.random(g.shape) >= model.dropout_probability)
    noise = noise_rng.normal(0, model.height_noise_std_m, g.shape)
    h, std = np.full(g.shape, np.nan), np.full(g.shape, np.nan)
    h[valid] = terrain.height_m[valid]+noise[valid]
    std[valid] = model.height_noise_std_m
    return Observation(g, h, valid, std, terrain.version, stamp_s, pose, model, seed)
