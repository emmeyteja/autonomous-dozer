# Autonomous Dozer

Simulation-first research project for sand-pile estimation and dozer pass planning.

## Project brief

Read the [project brief](docs/project-brief.pdf) for the proposed architecture, assumptions, implementation sequence, and open questions.

## Intended scope

- Estimate visible and hidden sand-pile geometry using lidar and camera observations, shape priors, and an optional mass constraint.
- Plan layered push passes using blade dimensions, capacity, overlap, and push-distance limits.
- Develop a CPU-based heightmap simulator before calibrating against higher-fidelity soil simulation.
- Re-scan and replan after each pass, and evaluate volume estimates and final grade error.

## Status

The first CPU simulation baseline is implemented: synthetic pile generation, a deterministic layer/lane material planner, capacity-limited transfers, repose relaxation, and replanning after each transfer. D61EX-24 and D61PX-24 profiles separate published dimensions from simulation assumptions.

A separate [partial-observation simulator](docs/observations.md) now models occlusion, range/FOV, noise and dropout. The material-planning demo still uses perfect simulated terrain and prescribed blade heights. It does not yet include perception, tracked vehicle motion, validated fallback supervision, hydraulic control, or a real Trimble connection. See [model limits and architecture](docs/architecture.md).

## Run the baseline

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,plots]'
python -m pytest -q
python -m autonomous_dozer --profile configs/komatsu_d61ex24_eu_stage_v.json --output outputs/d61-demo --plot
```

Choose a new output directory for every run. Use `--pile cone`, `--pile ridge`, or `--pile multi --seed 7`; use `--profile configs/komatsu_d61px24_eu_stage_v.json` to compare the PX variant. Plotting is optional; omit `--plot` and the `plots` extra for a NumPy-only runtime.

Outputs include a versioned pass history and metrics in `report.json`, initial/final arrays in `terrain.npz`, and an optional `comparison.png`. This scenario moves excess into a separate deposit region; successful upper-excess removal is not proof of general grading or vehicle performance.

See [manufacturer manuals, CAD access and dimension sources](docs/machine-references.md).

## Proposed implementation sequence

1. Heightmap simulator, pile generators, and repose-based slumping.
2. Visibility masking and lidar noise.
3. Pile-shape fitting and comparison against simulation ground truth.
4. Rule-based pass planning with a closed re-scan loop.
5. Evaluation across occlusion, density, noise, and volume uncertainty.
6. Soil-simulation calibration and synthetic sensors.
7. Optional path tracking and reinforcement-learning experiments.

The sequence remains the roadmap; only the initial material-simulation baseline and a simple deterministic planner are implemented. Simulation results require calibration and real-world validation before vehicle use.

## Architecture principles

- Reuse compatible Autoware interfaces wherever practical; justify custom interfaces when existing semantics do not fit.
- Keep estimation, planning, simulation, execution, and evaluation modular, with explicit contracts and dependencies.
- Keep core algorithms independent of ROS, simulator, and hardware integrations through replaceable adapters.
- Make resource limits and map resolution configurable so the system can scale as workloads grow.
- Start with a small modular baseline and justify added complexity with requirements or measurements.

These principles apply throughout development. See [development guidance](AGENTS.md).

## Development tracking

[GitHub milestones](https://github.com/emmeyteja/autonomous-dozer/milestones) and [issues](https://github.com/emmeyteja/autonomous-dozer/issues) are the canonical task outline, dependencies and acceptance criteria. Complete one issue at a time, record test evidence, and push tested commits directly to `main`. Pull requests are not part of the current development flow.
