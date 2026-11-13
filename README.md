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

Planning only. This repository currently contains the project brief and documentation; no simulator, estimator, planner, or vehicle-control implementation is included.

The brief proposes treating real-world Trimble blade control as an external system. Blade face height and target elevation need separate definitions before implementation. Other unresolved choices include blade specifications, spreading versus moving material to a dump zone, sand properties, and acceptance criteria.

## Proposed implementation sequence

1. Heightmap simulator, pile generators, and repose-based slumping.
2. Visibility masking and lidar noise.
3. Pile-shape fitting and comparison against simulation ground truth.
4. Rule-based pass planning with a closed re-scan loop.
5. Evaluation across occlusion, density, noise, and volume uncertainty.
6. Soil-simulation calibration and synthetic sensors.
7. Optional path tracking and reinforcement-learning experiments.

These are proposed milestones, not completed capabilities. Simulation results will require calibration and real-world validation before vehicle use.

## Architecture principles

- Reuse compatible Autoware interfaces wherever practical; justify custom interfaces when existing semantics do not fit.
- Keep estimation, planning, simulation, execution, and evaluation modular, with explicit contracts and dependencies.
- Keep core algorithms independent of ROS, simulator, and hardware integrations through replaceable adapters.
- Make resource limits and map resolution configurable so the system can scale as workloads grow.
- Start with a small modular baseline and justify added complexity with requirements or measurements.

These principles apply throughout development. See [development guidance](AGENTS.md).
