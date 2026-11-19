"""Deterministic next-pass selection over known simulation terrain."""
import math
import numpy as np
from .domain import Blade, Job, Rect, Terrain, WorkPass


class InfeasibleJob(ValueError):
    """Job is outside the supported simulation envelope."""


def next_pass(terrain: Terrain, job: Job, blade: Blade, overlap=0.15) -> WorkPass | None:
    """Recompute the top-down cut and select the largest-volume lane.

    Lane width is rounded DOWN to whole cells so it never exceeds blade width.
    Only +x cuts into an explicitly separate disposal area are implemented.
    This is material planning, not a collision-checked vehicle trajectory.
    """
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must lie in [0, 1)")
    g = terrain.grid
    if not g.contains(job.work) or not g.contains(job.deposit):
        raise InfeasibleJob("Work/deposit region lies outside map")
    work_mask = g.mask(job.work)
    deposit_mask = g.mask(job.deposit)
    if not work_mask.any() or not deposit_mask.any():
        raise InfeasibleJob("Work/deposit region is smaller than grid sampling")
    if job.deposit.xmin_m < job.work.xmax_m:
        raise InfeasibleJob("Initial planner requires disposal ahead of +x work region")
    if job.deposit.xmax_m-job.work.xmin_m > blade.max_push_m:
        raise InfeasibleJob("Conservative push-distance bound exceeds scenario limit")
    peak = float(np.max(terrain.height_m[work_mask]))
    if peak <= job.target_height_m+job.tolerance_m:
        return None
    cut_height = max(job.target_height_m, peak-0.75*blade.face_height_m)
    lane_cells = int(math.floor(blade.width_m/g.resolution_m))
    if lane_cells < 1:
        raise InfeasibleJob("Grid too coarse for blade width")
    step = max(1, int(math.floor(lane_cells*(1-overlap))))
    rows = np.flatnonzero(work_mask.any(axis=1))
    best = None
    best_volume = -1.0
    for row in range(int(rows[0]), int(rows[-1])+1, step):
        ymin = max(job.work.ymin_m, g.origin_y_m+row*g.resolution_m)
        ymax = min(job.work.ymax_m, g.origin_y_m+(row+lane_cells)*g.resolution_m)
        lane = Rect(job.work.xmin_m, job.work.xmax_m, ymin, ymax)
        volume = float(np.maximum(terrain.height_m-cut_height, 0)[g.mask(lane)].sum()*g.area_m2)
        if volume > best_volume:
            best, best_volume = lane, volume
    if best is None or best_volume <= 0:
        raise InfeasibleJob("No removable material in supported lanes")
    return WorkPass(terrain.version, best, job.deposit, cut_height, blade.effective_capacity_m3)


def metrics(terrain: Terrain, job: Job) -> dict:
    h = terrain.height_m[terrain.grid.mask(job.work)]
    if h.size == 0:
        raise ValueError("Job contains no grid cell centers")
    excess = np.maximum(h-job.target_height_m, 0)
    return {
        "volume_m3": terrain.volume_m3,
        "work_excess_m3": float(excess.sum()*terrain.grid.area_m2),
        "max_excess_m": float(excess.max()),
        "grade_rmse_m": float(np.sqrt(np.mean((h-job.target_height_m)**2))),
        "fraction_within_grade_tolerance": float(np.mean(np.abs(h-job.target_height_m) <= job.tolerance_m)),
    }
