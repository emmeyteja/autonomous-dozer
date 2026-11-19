# Estimator evaluation and promotion gates

[Issue #5](https://github.com/emmeyteja/autonomous-dozer/issues/5) adds reproducible synthetic experiments. This harness measures limitations; it never enables estimation-driven planning or changes estimator settings based on held-out outcomes.

## Run

```bash
python -m autonomous_dozer.evaluation_cli --split heldout --scenes-per-split 5 --output outputs/heldout-01
```

Use `--split all` for disjoint train/development/held-out geometry streams. `--scenes-per-split` is the number of independent repetitions per shape, not total cases. Each repetition contains cone, ridge, asymmetric and separated multi-dump fixtures, each with 13 profiles. Five held-out repetitions produce 260 cases, including 130 supported cone/ridge cases. A case-count budget prevents accidentally unbounded sweeps, and duplicate selections cannot inflate counts.

The profiles vary eye height/viewpoint, range, horizontal FOV, Gaussian height noise, dropout, mass availability and +/-15% assumed density bias. Low-eye-height density-bias combinations are included because density errors can matter most when geometry is occluded. This is a designed set of stress cases, not an exhaustive factorial experiment or a real-site distribution. Asymmetric/multi-dump stress fields are intentionally outside the fitted families and are not claimed to be settled physical piles.

Geometry streams are disjoint by split, repetition and shape. Sensor profiles share a geometry and random stream to support controlled comparisons. The synthetic mass ticket comes from independently specified continuous shape/material parameters; hidden raster heights are not passed to the estimator. Truth is used to generate observations and to score results only. There is no learning or automatic tuning in this harness; training and development partitions are available for future development.

## Reports and denominators

`evaluation.json` contains every case, source hashes, library versions, seeds, geometry/sensor/mass settings, fit configuration, the unchanged gate policy and summaries by profile and split. Fit runtime excludes observation generation. Failed fits and caught numerical exceptions remain records, never silently removed.

Metrics include total/visible/hidden volume relative errors, height RMSE over the union of true/predicted pile support, footprint IoU above the policy's height threshold, empirical conditional-interval coverage, rescan rate and recall on poor/missing point estimates. Relative error denominators have an explicit 0.01 m times cell-area volume floor near zero volume.

Quality success, rescan and missing-fit rates use all cases in the named group. Interval coverage over all cases counts missing or locally unidentifiable intervals as uncovered; coverage among available intervals is also reported with its denominator. Means over available point estimates report both available and missing counts. This avoids making failure cases disappear from accuracy rates. Rescan-on-good-point-fit is not automatically considered a supervisor error: a correct point estimate can still be structurally ambiguous.

The uncertainty intervals remain conditional research diagnostics. Empirical coverage measures them on these fixtures; it does not turn them into calibrated probability bounds. Rescan metrics are evaluated against truth-based point-fit quality, not a claim that all unknown-shape risks are observable.

## Predeclared gates

`configs/estimator_promotion_gates_v1.json` declares thresholds before evaluation: 10% total-volume error, 20% hidden-volume error, 0.08 m height RMSE and 0.75 footprint IoU define a quality pass. Promotion additionally requires at least 100 held-out supported cases, 90% quality success and interval coverage, no more than 50% supported rescans, and no more than 5% poor estimates among accepted held-out cases, including unsupported shapes. Train/development results cannot influence promotion checks.

A completed experiment exits zero even when promotion gates fail; inspect `promotion_checks` and `promotion_gates_passed`. Small CI smoke runs cannot establish promotion eligibility. `planning_enabled` is always false. Passing these synthetic research gates still requires broader distributions, calibration, adapter supervision and a separate explicit promotion decision.
