"""Generate partial observations without running an oracle planner.

Usage: python -m autonomous_dozer.observation_cli --output outputs/scan --plot
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import numpy as np
from .domain import Grid
from .terrain import pile, multi_dump
from .simulation import relax
from .observations import SensorPose, SensorModel, observe


def write_observation(observation, output: Path):
    """Write measurements only; truth is never embedded in this artifact."""
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"observation.npz", height_m=observation.height_m,
                        observed=observation.observed, height_std_m=observation.height_std_m)
    report = {
        "schema": "partial_height_observation_v1",
        "grid": asdict(observation.grid), "source_version": observation.source_version,
        "stamp_s": observation.stamp_s, "seed": observation.seed,
        "sensor_pose": asdict(observation.sensor_pose), "sensor_model": asdict(observation.sensor_model),
        "observed_cells": int(observation.observed.sum()),
        "unknown_cells": int((~observation.observed).sum()),
        "unknown_encoding": "NaN height and standard deviation; observed=false",
        "uncertainty": "independent vertical measurement-noise standard deviation only",
        "model": "ideal_cell_top_visibility_no_beam_pattern_or_vertical_fov",
        "planner_executed": False,
    }
    (output/"observation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/observation"))
    parser.add_argument("--pile", choices=["cone", "ridge", "multi"], default="ridge")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sensor-x", type=float, default=-8.0)
    parser.add_argument("--sensor-y", type=float, default=0.0)
    parser.add_argument("--sensor-z", type=float, default=2.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--fov-deg", type=float, default=120.0)
    parser.add_argument("--max-range", type=float, default=22.0)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--stamp", type=float, default=0.0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh directory")
    try:
        model = SensorModel(max_range_m=args.max_range, horizontal_fov_rad=math.radians(args.fov_deg),
                            height_noise_std_m=args.noise_std, dropout_probability=args.dropout)
        pose = SensorPose(args.sensor_x, args.sensor_y, args.sensor_z, math.radians(args.yaw_deg))
        # This small standard observation fixture is independent of the material planner.
        grid = Grid(96, 80, 0.25, -12, -10)
        raw = multi_dump(grid, seed=args.seed) if args.pile == "multi" else pile(
            grid, center=(-1, 0), radius_m=2.8, height_m=1.6,
            half_length_m=0.9 if args.pile == "ridge" else 0)
        settled = relax(raw)
        if not settled.converged:
            parser.error("Terrain failed to settle; no observation saved")
        observation = observe(settled.terrain, pose, model, seed=args.seed, stamp_s=args.stamp)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    report = write_observation(observation, args.output)
    if args.plot:
        from .visualization import plot_observation
        plot_observation(settled.terrain, observation, args.output/"observation.png")
    print(json.dumps({k: report[k] for k in ("observed_cells", "unknown_cells", "planner_executed")}, indent=2))
    print(f"Observation: {args.output/'observation.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
