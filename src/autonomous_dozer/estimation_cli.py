"""Fit a saved partial scan, without loading or executing the truth simulator."""
import argparse
from dataclasses import asdict
import json
from importlib.metadata import version
from pathlib import Path
import numpy as np
from .domain import Grid, Rect
from .observations import Observation, SensorModel, SensorPose
from .estimation import FitConfig, GroundReference, MassPrior, fit_pile


def load_observation(directory: Path) -> Observation:
    metadata = json.loads((directory/"observation.json").read_text())
    if metadata["schema"] != "partial_height_observation_v1":
        raise ValueError("Unsupported observation schema")
    with np.load(directory/"observation.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"height_m", "observed", "height_std_m"}:
            raise ValueError("Expected measurements only in observation artifact")
        return Observation(Grid(**metadata["grid"]), arrays["height_m"], arrays["observed"],
                           arrays["height_std_m"], metadata["source_version"], metadata["stamp_s"],
                           SensorPose(**metadata["sensor_pose"]), SensorModel(**metadata["sensor_model"]),
                           metadata["seed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-area", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--ground-elevation", type=float, default=0)
    parser.add_argument("--ground-std", type=float, default=0)
    parser.add_argument("--mass-kg", type=float)
    parser.add_argument("--density-kg-m3", type=float)
    parser.add_argument("--mass-relative-volume-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh directory")
    if (args.mass_kg is None) != (args.density_kg_m3 is None):
        parser.error("Mass and bulk density must be supplied together")
    try:
        observation = load_observation(args.observation)
        work = Rect(*args.work_area) if args.work_area else observation.grid.bounds
        ground = GroundReference(args.ground_elevation, args.ground_std)
        mass = MassPrior(args.mass_kg, args.density_kg_m3, args.mass_relative_volume_std) if args.mass_kg is not None else None
        config = FitConfig(seed=args.seed)
        result = fit_pile(observation, work, ground=ground, mass=mass, config=config)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    report = {
        "schema": "pile_estimate_v1", "library_versions": {"numpy": version("numpy"), "scipy": version("scipy")}, "status": result.status, "reasons": result.reasons,
        "rescan_required": result.rescan_required, "selected_model": result.selected_model,
        "source_version": result.source_version, "stamp_s": result.stamp_s, "frame_id": result.frame_id,
        "grid": asdict(observation.grid), "work_area": asdict(work), "ground_reference": asdict(ground),
        "mass_prior": asdict(mass) if mass else None, "fit_config": asdict(config),
        "parameters": result.parameters, "diagnostics": result.diagnostics,
        "candidates": result.candidates, "planner_executed": False,
        "provenance": "parametric_inference_from_measurements_only_not_observed_hidden_terrain",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"estimate.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if result.predicted_height_m is not None:
        np.savez_compressed(args.output/"estimate.npz", predicted_excess_height_m=result.predicted_height_m,
                            observed_mask=result.observed_mask, inferred_mask=~result.observed_mask)
        if args.plot:
            from .visualization import plot_estimate
            plot_estimate(observation, result, ground.elevation_m, args.output/"estimate.png")
    print(json.dumps({k: report[k] for k in ("status", "rescan_required", "selected_model", "reasons")}, indent=2))
    print(f"Estimate: {args.output/'estimate.json'}")
    return 2 if result.rescan_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
