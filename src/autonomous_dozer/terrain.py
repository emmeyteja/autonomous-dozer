"""Synthetic flat-ground piles; no perception or hidden-side estimator yet."""
import math
import numpy as np
from .domain import Grid, Terrain, positive


def pile(grid: Grid, *, center=(0.0, 0.0), radius_m=3.0, height_m=1.5,
         half_length_m=0.0, yaw_rad=0.0) -> Terrain:
    """Conical or stadium-footprint ridge pile, clipped at zero height."""
    positive("radius_m", radius_m)
    positive("height_m", height_m)
    if not all(math.isfinite(v) for v in (*center, half_length_m, yaw_rad)) or half_length_m < 0:
        raise ValueError("Invalid pile geometry")
    x, y = grid.centers()
    dx, dy = x-center[0], y-center[1]
    u = math.cos(yaw_rad)*dx + math.sin(yaw_rad)*dy
    v = -math.sin(yaw_rad)*dx + math.cos(yaw_rad)*dy
    distance = np.hypot(np.maximum(np.abs(u)-half_length_m, 0), v)
    return Terrain(grid, height_m*np.maximum(1-distance/radius_m, 0))


def multi_dump(grid: Grid, seed: int = 7, count: int = 3) -> Terrain:
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    rng = np.random.default_rng(seed)
    heights = np.zeros(grid.shape)
    for _ in range(count):
        p = pile(grid, center=tuple(rng.uniform(-1.0, 1.0, 2)),
                 radius_m=float(rng.uniform(1.0, 2.0)), height_m=float(rng.uniform(0.5, 1.0)))
        heights += p.height_m
    return Terrain(grid, heights)
