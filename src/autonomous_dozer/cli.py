"""Reproducible material-planning demo using perfect simulated terrain."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .domain import Grid, Job, Rect
from .terrain import pile, multi_dump
from .simulation import apply_pass, relax
from .planning import next_pass, metrics
from .profiles import load_blade


def run_demo(blade, *, pile_type="ridge", seed=7, resolution_m=0.25,
             max_passes=100, target_height_m=0.0):
    if type(max_passes) is not int or max_passes < 1:
        raise ValueError("max_passes must be a positive integer")
    # Keep bounds fixed for comparable scenario metrics at different resolutions.
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution_m must be finite and positive")
    nx, ny = round(24/resolution_m), round(20/resolution_m)
    if (not np.isclose(nx*resolution_m, 24) or not np.isclose(ny*resolution_m, 20)
            or nx*ny > 1_000_000):
        raise ValueError("Resolution must tile 24 x 20 m and use <= 1,000,000 cells")
    grid = Grid(nx, ny, resolution_m, -12, -10)
    job = Job(Rect(-6, 4, -5, 5), Rect(7, 11, -4, 4), target_height_m)
    if pile_type == "multi":
        initial = multi_dump(grid, seed=seed)
    elif pile_type in {"cone", "ridge"}:
        initial = pile(grid, center=(-1, 0), radius_m=2.8, height_m=1.6,
                       half_length_m=0.9 if pile_type == "ridge" else 0)
    else:
        raise ValueError("Unknown pile type")
    settled = relax(initial)
    terrain = settled.terrain
    history = []
    status = "pass_limit"
    if not settled.converged:
        status = "relaxation_limit"
    else:
        for _ in range(max_passes):
            command = next_pass(terrain, job, blade)
            if command is None:
                status = "upper_excess_within_tolerance"
                break
            transfer = apply_pass(terrain, command)
            settled = relax(transfer.terrain)
            terrain = settled.terrain
            history.append({"pass": len(history)+1, "command": asdict(command),
                            "moved_m3": transfer.moved_m3, "relaxation_sweeps": settled.sweeps,
                            "relaxation_converged": settled.converged, **metrics(terrain, job)})
            if not settled.converged:
                status = "relaxation_limit"
                break
            if transfer.moved_m3 <= 1e-10:
                status = "no_progress"
                break
        else:
            if next_pass(terrain, job, blade) is None:
                status = "upper_excess_within_tolerance"
    summary = {
        "status": status, "model": "macroscopic_heightmap_transfer",
        "observation": "perfect_simulation_terrain_no_sensor_estimator",
        "goal": "remove_excess_from_work_region_to_separate_deposit_region",
        "blade_authority": "prescribed_simulation_only",
        "pile_type": pile_type, "seed": seed, "passes": len(history),
        "grid": asdict(grid), "job": asdict(job), "blade": asdict(blade),
        "initial": metrics(initial, job), "final": metrics(terrain, job),
        "mass_balance_error_m3": terrain.volume_m3-initial.volume_m3,
        "history": history,
    }
    return initial, terrain, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    p.add_argument("--pile", choices=["cone", "ridge", "multi"], default="ridge")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resolution", type=float, default=0.25)
    p.add_argument("--max-passes", type=int, default=100)
    p.add_argument("--target-height", type=float, default=0.0)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output already exists; choose a fresh directory")
    try:
        initial, final, report = run_demo(load_blade(args.profile), pile_type=args.pile,
                                        seed=args.seed, resolution_m=args.resolution,
                                        max_passes=args.max_passes, target_height_m=args.target_height)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))
    report["machine_profile"] = json.loads(args.profile.read_text())
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/"terrain.npz", initial=initial.height_m, final=final.height_m)
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if args.plot:
        from .visualization import plot_demo
        plot_demo(initial, final, report, args.output/"comparison.png")
    print(json.dumps({k: report[k] for k in ("status", "passes", "mass_balance_error_m3", "final")}, indent=2))
    print(f"Report: {args.output/'report.json'}")
    return 0 if report["status"] == "upper_excess_within_tolerance" else 2
