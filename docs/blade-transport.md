# Blade carrying and deposition

`HeightmapTransport` implements the `TransportModel` domain adapter in `transport.py`. It accepts immutable `Terrain`, `Stroke`, `Blade` and `CarryState` values and returns new terrain, final carried load, sample accounting, limitations and grade metrics. It imports no ROS, Autoware, simulator or hardware SDK. No new ROS interface is introduced; future integrations must convert compatible standard/Autoware messages at their boundaries.

Run from the repository root:

```bash
python -m autonomous_dozer.transport_cli --output outputs/transport-fill
python -m autonomous_dozer.transport_cli --objective move --shedding-fraction-per-m 0.1 --output outputs/transport-move
```

Choose a fresh output directory. JSON records the full synthetic configuration, per-sample material accounting and result; NPZ contains before/after heights. Exit zero means the experiment completed. Read `objective_complete` and `flags` to determine outcome. This is one stroke, not a completed job. The demo uses a synthetic blade, not an OEM machine calibration.

## Model contract

- Coordinates are meters in the terrain grid frame. Volumes are cubic meters of unchanged bulk material; density and compaction are not modeled. Carry state must be nonnegative and no larger than effective blade capacity.
- Supported paths are straight, strictly increasing +x samples at constant y. Every crossed grid-column center is visited, including between sparse input samples. Lateral footprint must stay in the map and total stroke length must respect the configured push limit. These checks do not establish vehicle or swept-footprint feasibility.
- Only cells wholly inside the blade lane and cut/release rectangles participate. Subcell overlap is discarded, never expanded. Paths/regions with no full cut or release cells are rejected explicitly. Remaining-cut and grade metrics cover all full cells in their requested rectangles, including outside the blade lane.
- Each column cuts down toward prescribed `cut_height_m`, limited by `max_cut_depth_m`, free blade capacity and reserved release capacity along this stroke. Depth cannot exceed blade face height. Target grade is a separate parameter. Fully blocked disposal prevents new excavation; source terrain may still slump afterward.
- Material remains in carried state until local release cells are reached. Release is distributed in proportion to available height headroom at that column, never teleported to remote cells. Initial carried material that cannot fit remains carried and produces `disposal_full`; callers must pass that state into subsequent strokes.
- Optional shedding uses the uncalibrated exponential fraction `1-exp(-rate * distance)` of current load per column interval. It places material in available cells immediately outside the two blade shoulders, capped at the deposit-height limit. Missing/full shoulder cells retain material and report `shedding_blocked`. It is a coarse model of loss from the blade, not a hydraulic/soil model.
- Closed-boundary repose relaxation runs after the stroke. Terrain plus carried volume is conserved through all operations. Steps report mass residuals; the final independently summed residual is checked. Nonconvergence is explicit and prevents completion. Relaxation may move material beyond the release rectangle or above placement limits; `post_slump_deposit_limit_exceeded` reports the latter without destroying excess material.

## Objectives and limits

`move` requires all full cells in the cut rectangle to reach the cut floor. `fill` requires the entire release rectangle to reach target grade within tolerance from below. `finish` additionally checks the upper tolerance. Fill/finish placement is capped at target height, which cannot exceed the deposit-height limit. Preexisting overfill is reported; this single stroke cannot excavate its release area. All objectives also require empty carried load, converged relaxation and no post-slump deposit-height violation. Fill/finish can complete while source material remains: that is intentional, not a whole-source removal claim.

`capacity_limited`, `disposal_limited`, `depth_limited`, `blocked_disposal`, `disposal_full`, `no_work` and `grade_incomplete` describe constraints encountered; flags can coexist with a completed grade objective. `fill_deficit_m3` and `overfill_m3` are measured after slumping. `spill_m3` is positive net height increase outside the intended release rectangle, including source/shoulder slumping. It is an occupancy metric, not a material-provenance tracer; gross shed volume is separately available in the step trace.

The original `simulation.apply_pass` is retained as `simulation.apply_uniform_transfer`, the named fast baseline. The original planning CLI still uses that baseline. The path-based adapter is independently runnable and tested; it has not been wired into a trajectory planner or a closed perception loop. All cutting, carrying, shedding and repose parameters remain **uncalibrated**. Effective blade capacity is an experimental assumption, even when other machine dimensions come from an OEM profile.

Memory is proportional to the configured raster; path work is bounded by its column count. Per-column full-raster accounting deliberately favors independently checkable diagnostics in this CPU baseline. Large-map tiling and incremental accounting need measurement before deployment at larger scales.

External Trimble blade authority with dual GNSS remains the intended hardware boundary. This model issues no machine commands and makes no piston-feedback, dynamics, automatic fallback or safety-validation claims. Follow-up #7 addresses feasible earthwork passes and #8–#10 address execution/interfaces/supervision.
