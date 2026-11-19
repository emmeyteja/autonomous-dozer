# Partial terrain observation simulator

Implemented for [issue #3](https://github.com/emmeyteja/autonomous-dozer/issues/3). This is an idealized visibility/noise model, independent of ROS and of the material planner. It is not a calibrated LiDAR or camera simulator.

## Run

From the repository root after installing the project:

```bash
python -m autonomous_dozer.observation_cli --output outputs/scan-01 --plot
```

Select `--pile cone|ridge|multi`, `--seed`, sensor XYZ, `--yaw-deg`, `--fov-deg`, `--max-range`, `--noise-std`, `--dropout`, and simulated `--stamp` as needed. Use a fresh output directory; the command refuses to overwrite previous artifacts. The standard CLI scene uses a 24 x 20 m grid at 0.25 m resolution. The Python API accepts any valid `Grid` within the explicitly configured ray-cell budget.

## Observation contract

`Observation` contains grid/frame metadata, sensor pose/model, source terrain revision, acquisition time in simulation seconds, random seed, measured heights, a boolean observed mask, and vertical noise standard deviations. Arrays are copied and read-only by default. Source truth is never embedded in the contract.

Observed zero height means measured ground. Unknown height and uncertainty are NaN with `observed=false`; no-return, occlusion, field-of-view/range exclusion and dropout all remain unknown. Heights may be negative after Gaussian noise on flat ground: clipping to zero would introduce bias. The standard deviation describes the configured independent vertical measurement noise, not full terrain confidence, pose uncertainty, hidden-side uncertainty or calibration accuracy.

`observation.npz` contains only `height_m`, `observed`, and `height_std_m` and loads with `allow_pickle=False`. `observation.json` records metadata, units through field names, counts, model limitations and `planner_executed=false`. Optional `observation.png` compares separate truth with measurements for diagnostics only; do not use the diagnostic truth panel as an estimator input.

## Visibility and degradation

For each candidate cell-top center, range is measured in 3D, and horizontal bearing is compared with yaw and horizontal FOV. With no horizontal displacement the ray is treated as nadir and included regardless of yaw. The sensor must be inside the known grid and above its local terrain.

Ray/grid traversal tests intersections with piecewise-flat terrain columns. Intervening surfaces block a ray when they protrude above it. The target top must also be reachable without entering its own solid column. Exact zero-length corner touches are not occluders. This column model has grid-dependent staircase effects; it cannot represent overhangs, vertical-face returns, vegetation, machine self-occlusion, multiple returns or outside-map obstacles.

There is no vertical FOV or angular beam-sampling pattern. Every eligible visible cell top is a potential measurement. Gaussian noise is applied to height, not to raw beam range. Bernoulli dropout is independent of geometry/material. Independent seeded streams make common-cell noise stable when dropout alone changes. Reproduction assumes the same NumPy random-generator implementation/version.

The ray-cell budget is checked before tracing using a conservative sum of Manhattan cell-traversal bounds. Large maps/ranges fail explicitly rather than starting an unbounded ray loop. Regional/incremental scans and more realistic beam sampling are later improvements.

## Planning and validation boundaries

The material demo remains explicitly oracle-based and still reports `perfect_simulation_terrain_no_sensor_estimator`. `next_pass` rejects `Observation` inputs: unknown cells cannot silently become flat ground or enter the existing planner as complete terrain. Issue #4 adds the estimator; issue #5 measures estimation quality and confidence before observation-driven planning is enabled.

Tests cover flat ground, wall and diagonal occlusion, mirrored viewpoints, target-top self-occlusion, sensor placement, slant range/FOV, reproducibility, unbiased noise, total dropout, metadata, array ownership, exports, resource limits, and refusal of partial scans by the oracle planner.
