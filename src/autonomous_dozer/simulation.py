"""Mass-conserving macroscopic transfers and cardinal-neighbor repose relaxation.

This is a heightmap approximation, not soil mechanics or a moving blade solver.
"""
from dataclasses import dataclass
import math
import numpy as np
from .domain import Terrain, WorkPass, positive


@dataclass(frozen=True)
class Relaxation:
    terrain: Terrain
    sweeps: int
    converged: bool
    max_slope_excess_m: float


def relax(terrain: Terrain, angle_deg=32.0, max_sweeps=1000, tolerance_m=1e-5) -> Relaxation:
    """Pairwise checkerboard relaxation with closed (no-flux) map boundaries."""
    if not math.isfinite(angle_deg) or not 0 < angle_deg < 90:
        raise ValueError("Repose angle must lie in (0, 90) degrees")
    if type(max_sweeps) is not int or max_sweeps < 1:
        raise ValueError("max_sweeps must be positive")
    positive("tolerance_m", tolerance_m)
    h = terrain.height_m.copy()
    limit = math.tan(math.radians(angle_deg))*terrain.grid.resolution_m
    excess = math.inf
    for sweep in range(1, max_sweeps+1):
        # Each substep consists of disjoint pairs; transfers conserve material.
        for axis in (0, 1):
            for parity in (0, 1):
                first, second = [slice(None)]*2, [slice(None)]*2
                first[axis] = slice(parity, h.shape[axis]-1, 2)
                second[axis] = slice(parity+1, h.shape[axis], 2)
                a, b = h[tuple(first)], h[tuple(second)]
                diff = a-b
                transfer = 0.5*np.sign(diff)*np.maximum(np.abs(diff)-limit, 0)
                a -= transfer
                b += transfer
        excess = max(0.0, *(float(np.max(np.abs(np.diff(h, axis=a))))-limit for a in (0, 1)))
        if excess <= tolerance_m:
            break
    return Relaxation(Terrain(terrain.grid, h, terrain.version+1), sweep,
                      excess <= tolerance_m, excess)


@dataclass(frozen=True)
class Transfer:
    terrain: Terrain
    moved_m3: float


def apply_pass(terrain: Terrain, work_pass: WorkPass) -> Transfer:
    """Remove in +x cell-column order, capacity-limit, and place ahead uniformly.

    A partial final column models stopping within one grid cell. No material leaves
    the grid. Deposit shape and carrying dynamics are intentionally simplified.
    """
    p, grid = work_pass, terrain.grid
    if p.source_version != terrain.version:
        raise ValueError("Stale work pass: terrain has changed")
    if p.direction != "+x":
        raise ValueError("Only +x material transfers are implemented")
    positive("capacity_m3", p.capacity_m3)
    if not math.isfinite(p.cut_height_m) or p.cut_height_m < 0:
        raise ValueError("Invalid prescribed cut height")
    if not grid.contains(p.cut) or not grid.contains(p.deposit) or p.cut.overlaps(p.deposit):
        raise ValueError("Cut/deposit regions must be disjoint and inside the grid")
    if p.deposit.xmin_m < p.cut.xmax_m:
        raise ValueError("Deposit region must be ahead of the cut for a +x push")
    cut_mask, deposit_mask = grid.mask(p.cut), grid.mask(p.deposit)
    if not cut_mask.any() or not deposit_mask.any():
        raise ValueError("Cut/deposit region contains no grid cell centers")
    h = terrain.height_m.copy()
    remaining = p.capacity_m3
    for col in np.flatnonzero(cut_mask.any(axis=0)):
        available = np.where(cut_mask[:, col], np.maximum(h[:, col]-p.cut_height_m, 0), 0)
        column_volume = float(available.sum()*grid.area_m2)
        if column_volume > 0:
            fraction = min(1.0, remaining/column_volume)
            h[:, col] -= available*fraction
            remaining = max(0.0, remaining-column_volume*fraction)
        if remaining <= 1e-12:
            break
    # Compute removed volume from the grid, not capacity subtraction, for accounting.
    moved = float((terrain.height_m-h).sum()*grid.area_m2)
    h[deposit_mask] += moved/(np.count_nonzero(deposit_mask)*grid.area_m2)
    return Transfer(Terrain(grid, h, terrain.version+1), moved)


# Explicit baseline name for comparisons with path-based transport.
apply_uniform_transfer = apply_pass
