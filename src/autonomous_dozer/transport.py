"""Uncalibrated, closed-domain heightmap blade transport; no hydraulic dynamics."""
from dataclasses import dataclass
import math
from typing import Protocol
import numpy as np
from .domain import Blade, Rect, Terrain
from .simulation import Relaxation, relax


@dataclass(frozen=True)
class CarryState:
    volume_m3: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.volume_m3) or self.volume_m3 < 0:
            raise ValueError("Carried volume must be finite and nonnegative")


@dataclass(frozen=True)
class Stroke:
    source_version: int
    path_xy_m: tuple[tuple[float, float], ...]
    cut: Rect
    release: Rect
    cut_height_m: float = 0.0
    max_cut_depth_m: float = 0.2
    max_deposit_height_m: float = 1.0
    objective: str = "move"
    target_height_m: float = 0.1
    tolerance_m: float = 0.03
    shedding_fraction_per_m: float = 0.0

    def __post_init__(self):
        path = tuple(tuple(p) for p in self.path_xy_m)
        object.__setattr__(self, "path_xy_m", path)
        if (len(path) < 2 or any(len(p) != 2 for p in path)
                or not all(math.isfinite(v) for p in path for v in p)):
            raise ValueError("Need at least two finite XY path samples")
        if any(b[0] <= a[0] or b[1] != a[1] for a, b in zip(path, path[1:])):
            raise ValueError("Only straight increasing +x strokes are implemented")
        if type(self.source_version) is not int or self.source_version < 0:
            raise ValueError("Invalid source version")
        for name in ("cut_height_m", "target_height_m", "shedding_fraction_per_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("max_cut_depth_m", "max_deposit_height_m", "tolerance_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.objective not in ("move", "fill", "finish"):
            raise ValueError("Objective must be move, fill or finish")
        if self.objective != "move" and self.target_height_m > self.max_deposit_height_m:
            raise ValueError("Target grade exceeds deposit height limit")


@dataclass(frozen=True)
class TransportStep:
    x_m: float
    cut_m3: float
    released_m3: float
    shed_m3: float
    carried_m3: float
    mass_error_m3: float


@dataclass(frozen=True)
class TransportResult:
    terrain: Terrain
    carry: CarryState
    steps: tuple[TransportStep, ...]
    flags: tuple[str, ...]
    objective_complete: bool
    remaining_cut_m3: float
    fill_deficit_m3: float
    overfill_m3: float
    spill_m3: float
    mass_error_m3: float
    relaxation: Relaxation
    calibrated: bool = False


class TransportModel(Protocol):
    def apply(self, terrain: Terrain, stroke: Stroke, blade: Blade,
              carry: CarryState = CarryState()) -> TransportResult: ...


class HeightmapTransport:
    """One straight stroke, full-cell footprint, bounded placement, then slumping.

    Path is resampled at every crossed column. Excavation reserves space along
    the future release path; a blocked destination cannot trigger new digging.
    """
    def __init__(self, repose_angle_deg=32.0, max_relax_sweeps=1000):
        self.repose_angle_deg = repose_angle_deg
        self.max_relax_sweeps = max_relax_sweeps

    def apply(self, terrain, stroke, blade, carry=CarryState()):
        s, g = stroke, terrain.grid
        if s.source_version != terrain.version:
            raise ValueError("Stale stroke")
        if carry.volume_m3 > blade.effective_capacity_m3:
            raise ValueError("Initial carried load exceeds blade capacity")
        if s.max_cut_depth_m > blade.face_height_m:
            raise ValueError("Cut depth exceeds blade face height")
        if (not g.contains(s.cut) or not g.contains(s.release)
                or s.cut.overlaps(s.release) or s.release.xmin_m < s.cut.xmax_m):
            raise ValueError("Cut/release must be disjoint, inside map and ordered +x")
        start, end = s.path_xy_m[0], s.path_xy_m[-1]
        b, half = g.bounds, blade.width_m/2
        if (start[0] < b.xmin_m or end[0] > b.xmax_m
                or start[1]-half < b.ymin_m or start[1]+half > b.ymax_m):
            raise ValueError("Blade path footprint crosses map boundary")
        if end[0]-start[0] > blade.max_push_m:
            raise ValueError("Stroke exceeds configured maximum push distance")
        x, y = g.centers()
        r = g.resolution_m/2
        lane = (y-r >= start[1]-half-1e-10) & (y+r <= start[1]+half+1e-10)
        traversed = (x >= start[0]) & (x <= end[0])
        def cells(region):
            return ((x-r >= region.xmin_m-1e-10) & (x+r <= region.xmax_m+1e-10)
                    & (y-r >= region.ymin_m-1e-10) & (y+r <= region.ymax_m+1e-10))
        cut_region, release_region = cells(s.cut), cells(s.release)
        cut, release = cut_region & lane & traversed, release_region & lane & traversed
        if not cut.any() or not release.any():
            raise ValueError("Resolution/path leaves no full cut or release cells")
        h = terrain.height_m.copy()
        area, load = g.area_m2, carry.volume_m3
        initial_mass = terrain.volume_m3+load
        cap = blade.effective_capacity_m3
        ceiling = s.max_deposit_height_m if s.objective == "move" else s.target_height_m
        flags, steps = set(), []
        source = float(np.maximum(h-s.cut_height_m, 0)[cut].sum()*area)
        space = float(np.maximum(ceiling-h, 0)[release].sum()*area)
        if source > 1e-10 and space <= 1e-10:
            flags.add("blocked_disposal")
        def place(mask, amount, limit):
            room = np.where(mask, np.maximum(limit-h, 0), 0)
            volume = float(room.sum()*area)
            moved = min(amount, volume)
            if volume > 0:
                h[:] += room*(moved/volume)
            return moved
        rows = np.flatnonzero(lane[:, 0])
        previous_x = start[0]
        for col in np.flatnonzero(traversed.any(axis=0)):
            cx = float(x[0, col])
            space = float(np.maximum(ceiling-h, 0)[release].sum()*area)
            available = np.where(cut[:, col],
                                 np.minimum(np.maximum(h[:, col]-s.cut_height_m, 0),
                                            s.max_cut_depth_m), 0)
            volume = float(available.sum()*area)
            take = min(volume, max(0, cap-load), max(0, space-load))
            if volume > take+1e-10:
                flags.add("capacity_limited" if cap-load < space-load else "disposal_limited")
            if volume > 0:
                h[:, col] -= available*(take/volume)
            load += take
            shoulder = np.zeros(g.shape, dtype=bool)
            for row in (rows[0]-1, rows[-1]+1):
                if 0 <= row < g.ny:
                    shoulder[row, col] = True
            requested = load*(-math.expm1(-s.shedding_fraction_per_m*(cx-previous_x)))
            shed = place(shoulder, requested, s.max_deposit_height_m)
            if requested-shed > 1e-10:
                flags.add("shedding_blocked")
            load -= shed
            local = np.zeros(g.shape, dtype=bool)
            local[:, col] = release[:, col]
            released = place(local, load, ceiling)
            load = max(0.0, load-released)
            error = float(h.sum()*area+load-initial_mass)
            steps.append(TransportStep(cx, take, released, shed, load, error))
            previous_x = cx
        if load > 1e-10:
            flags.add("disposal_full")
        settled = relax(Terrain(g, h, terrain.version+1), self.repose_angle_deg,
                        self.max_relax_sweeps)
        final = settled.terrain
        if np.any(final.height_m[release_region] > s.max_deposit_height_m+1e-10):
            flags.add("post_slump_deposit_limit_exceeded")
        if not settled.converged:
            flags.add("relaxation_incomplete")
        remaining = float(np.maximum(final.height_m-s.cut_height_m, 0)[cut_region].sum()*area)
        # Grade goals cover the entire requested release region, including areas
        # outside this stroke's lane; one lane must not claim whole-job success.
        grade = release_region
        deficit = float(np.maximum(s.target_height_m-final.height_m, 0)[grade].sum()*area)
        overfill = float(np.maximum(final.height_m-s.target_height_m, 0)[grade].sum()*area)
        delta = final.height_m-s.target_height_m
        complete = (remaining <= 1e-10 if s.objective == "move" else
                    bool(np.all(delta[grade] >= -s.tolerance_m)))
        if s.objective == "finish":
            complete = complete and bool(np.all(delta[grade] <= s.tolerance_m))
        complete = (complete and load <= 1e-10 and settled.converged
                    and "post_slump_deposit_limit_exceeded" not in flags)
        if source <= 1e-10 and carry.volume_m3 <= 1e-10:
            flags.add("no_work")
        if remaining > 1e-10 and np.any((terrain.height_m-s.cut_height_m)[cut] > s.max_cut_depth_m):
            flags.add("depth_limited")
        if s.objective != "move" and not complete:
            flags.add("grade_incomplete")
        spill = float(np.maximum(final.height_m-terrain.height_m, 0)[~release_region].sum()*area)
        error = final.volume_m3+load-initial_mass
        if abs(error) > 1e-8*max(1, initial_mass):
            raise RuntimeError("Transport mass balance failed")
        return TransportResult(final, CarryState(load), tuple(steps), tuple(sorted(flags)),
                               complete, remaining, deficit, overfill, spill, error, settled)
