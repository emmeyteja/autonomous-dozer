"""Generate a reproducible estimator sweep report without promoting the estimator."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import hashlib
from pathlib import Path
from .estimation import FitConfig
from .evaluation import PROFILES, SHAPES, SPLITS, run_sweep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gates", type=Path, default=Path("configs/estimator_promotion_gates_v1.json"))
    p.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    p.add_argument("--scenes-per-split", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a fresh output directory")
    gates = json.loads(args.gates.read_text())
    config = FitConfig()
    try:
        rows, summary = run_sweep(gates=gates, splits=tuple(SPLITS) if args.split == "all" else (args.split,),
                                  scenes_per_split=args.scenes_per_split, root_seed=args.seed, fit_config=config)
    except (ValueError, TypeError) as exc:
        p.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "estimator_evaluation_v1", "root_seed": args.seed, "fit_config": asdict(config),
              "gates": gates, "scenario_generator_version": "v1",
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name+".py").read_bytes()).hexdigest()
                                for name in ("evaluation", "estimation", "observations", "terrain")},
              "library_versions": {"numpy": version("numpy"), "scipy": version("scipy")},
              "runtime_scope": "fit_only_seconds_excludes_scan_simulation", "shapes": SHAPES, "profile_definitions": PROFILES, "summary": summary, "cases": rows}
    (args.output/"evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"cases": len(rows), "promotion_checks": summary["promotion_checks"], "planning_enabled": False}, indent=2))
    return 0  # The experiment completed; failed promotion gates are a result, not a crash.


if __name__ == "__main__":
    raise SystemExit(main())
