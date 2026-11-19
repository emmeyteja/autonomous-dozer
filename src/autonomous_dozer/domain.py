"""Explicit SI-unit contracts shared by planning and simulation."""
from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray


def positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Rect:
    xmin_m: float
    xmax_m: float
    ymin_m: float
    ymax_m: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in self.__dict__.values()):
            raise ValueError("Rectangle coordinates must be finite")
        if self.xmin_m >= self.xmax_m or self.ymin_m >= self.ymax_m:
            raise ValueError("Rectangle must have positive area")

    def overlaps(self, other: "Rect") -> bool:
        return (self.xmin_m < other.xmax_m and other.xmin_m < self.xmax_m
                and self.ymin_m < other.ymax_m and other.ymin_m < self.ymax_m)


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    resolution_m: float
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0
    frame_id: str = "site"

    def __post_init__(self):
        if type(self.nx) is not int or type(self.ny) is not int or min(self.nx, self.ny) < 2:
            raise ValueError("Grid dimensions must be integers >= 2")
        positive("resolution_m", self.resolution_m)
        if not all(math.isfinite(v) for v in (self.origin_x_m, self.origin_y_m)) or not self.frame_id:
            raise ValueError("Grid needs finite origin and a frame")

    @property
    def shape(self):
        return self.ny, self.nx

    @property
    def area_m2(self):
        return self.resolution_m ** 2

    @property
    def bounds(self):
        return Rect(self.origin_x_m, self.origin_x_m + self.nx*self.resolution_m,
                    self.origin_y_m, self.origin_y_m + self.ny*self.resolution_m)

    def centers(self):
        return np.meshgrid(self.origin_x_m + (np.arange(self.nx)+0.5)*self.resolution_m,
                           self.origin_y_m + (np.arange(self.ny)+0.5)*self.resolution_m)

    def mask(self, region: Rect):
        x, y = self.centers()
        return ((x >= region.xmin_m) & (x < region.xmax_m)
                & (y >= region.ymin_m) & (y < region.ymax_m))

    def contains(self, region: Rect):
        b = self.bounds
        return (b.xmin_m <= region.xmin_m < region.xmax_m <= b.xmax_m
                and b.ymin_m <= region.ymin_m < region.ymax_m <= b.ymax_m)


@dataclass(frozen=True)
class Terrain:
    grid: Grid
    height_m: NDArray[np.float64]
    version: int = 0

    def __post_init__(self):
        h = np.array(self.height_m, dtype=float, copy=True)
        if h.shape != self.grid.shape or not np.all(np.isfinite(h)) or np.any(h < 0):
            raise ValueError("Terrain must be a finite nonnegative height-above-flat-ground grid")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("Terrain version must be a nonnegative integer")
        h.flags.writeable = False
        object.__setattr__(self, "height_m", h)

    @property
    def volume_m3(self):
        return float(self.height_m.sum()*self.grid.area_m2)


@dataclass(frozen=True)
class Blade:
    width_m: float
    face_height_m: float
    rated_capacity_m3: float
    capacity_factor: float = 0.5  # Experimental effective fill, not OEM capacity calibration.
    max_push_m: float = 20.0     # Scenario assumption, not an OEM limit.

    def __post_init__(self):
        for name in ("width_m", "face_height_m", "rated_capacity_m3", "max_push_m"):
            positive(name, getattr(self, name))
        if not math.isfinite(self.capacity_factor) or not 0 < self.capacity_factor <= 1:
            raise ValueError("capacity_factor must be in (0, 1]")

    @property
    def effective_capacity_m3(self):
        return self.rated_capacity_m3*self.capacity_factor


@dataclass(frozen=True)
class Job:
    work: Rect
    deposit: Rect
    target_height_m: float
    tolerance_m: float = 0.03

    def __post_init__(self):
        if not math.isfinite(self.target_height_m) or self.target_height_m < 0:
            raise ValueError("target_height_m must be finite and nonnegative")
        positive("tolerance_m", self.tolerance_m)
        if self.work.overlaps(self.deposit):
            raise ValueError("Work and deposit regions must not overlap")


@dataclass(frozen=True)
class WorkPass:
    source_version: int
    cut: Rect
    deposit: Rect
    cut_height_m: float
    capacity_m3: float
    # Initial material-transfer model supports straight +x pushes only.
    direction: str = "+x"
