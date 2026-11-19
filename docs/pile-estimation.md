# Partial-observation pile estimation

Implemented for [issue #4](https://github.com/emmeyteja/autonomous-dozer/issues/4). This module fits a single isolated pile above a supplied horizontal ground reference. It consumes `Observation`, a known work area, and an optional mass/density prior. It never consumes `Terrain` or hidden simulation arrays. Planning remains separate.

## Run on a saved scan

Install the optional numerical solver, create a scan as documented in [observations](observations.md), then fit it:

```bash
python -m pip install -e '.[estimation,plots]'
python -m autonomous_dozer.estimation_cli --observation outputs/scan-01 --output outputs/fit-01 --work-area -6 5 -5 5 --plot
```

Exit code 0 means `estimated` under the configured research gates. Exit code 2 means rescan required or invalid CLI input. A completed but rejected fit still writes its diagnostic report; insufficient data or optimization failure produces no predicted-height artifact. Existing output directories are not overwritten.

To supply an independent weighbridge/material prior, add both `--mass-kg` and `--density-kg-m3`, with `--mass-relative-volume-std` expressing combined volume uncertainty. Never derive an operational prior from the same hidden ground truth being estimated. Synthetic tests may specify independent scenario tickets. For example, the standard ridge fixture has an approximately 31,800 kg synthetic ticket at 1,500 kg/m³; these numbers are not a real measurement or calibrated density.

## Model and fit

Both hypotheses use positive excess height above the supplied ground elevation:

- Cone: `h = slope * max(radius - distance_to_center, 0)`.
- Ridge: distance is measured to an oriented centerline segment with half-length `a`, producing a stadium footprint and rounded ends.

Cone parameters are center XY, radius and slope; the ridge additionally fits half-length and yaw. Slope is bounded to 25-40 degrees, with a weak 32 +/- 7.5 degree prior. No independent plateau/cap parameter is fitted: capped piles, compound dumps and arbitrary asymmetric shapes can violate these hypotheses.

The solver uses [SciPy bounded least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html), with deterministic multistart initialization. Data residuals receive a Huber transform; free-ground, boundary, slope and mass terms remain quadratic. Only observed cells contribute data or free-ground penalties. Approximately flat measured cells constrain predicted excess above a three-sigma ground band. Unknown cells are never treated as flat ground.

The known work area constrains the footprint through boundary penalties and a post-fit boundary gate. Candidate comparison evaluates all known cells in that area, not just the bounded optimization sample. A parameter-count penalty favors simpler near-equivalent shapes. An explicit start probes a ridge extending away from the sensor behind the observed end cap, which helps reveal cone-versus-hidden-ridge ambiguity.

`FitConfig` bounds starts, evaluations, sample count and grid size. The default samples at most 1,200 measurements, uses six starts per hypothesis and at most 250 solver evaluations per start. SciPy's evaluation count excludes finite-difference Jacobian calls; this is a bounded research workload, not a realtime guarantee. Seed and library versions should accompany comparisons.

## Results and provenance

`estimate.json` records status, rescan reasons, selected hypothesis, parameters, source terrain revision/time/frame, ground/mass/configuration inputs, all converged candidate summaries, residual and uncertainty diagnostics. `estimate.npz`, when available, contains:

- `predicted_excess_height_m`: a model prediction above supplied ground, including the hidden side; it is not a measured or ground-truth height map.
- `observed_mask`: the original measurement-validity mask.
- `inferred_mask`: its complement. This marks where no measurements were available, not an independently validated inference region.

Measured cells also have model predictions; the original observation file remains authoritative for raw measurements. The estimate is not a `Terrain` snapshot and cannot be passed directly to the oracle planner.

Total, visible and hidden volumes are numerical integrals over the output grid, with visible/hidden partitioned by the original observation mask. A separate analytic full-footprint volume is used for the mass prior and continuous-volume uncertainty diagnostics; raster and analytic values differ slightly with resolution. For a ridge this analytic volume is `slope * (pi*radius^3/3 + 2*a*radius^2)`.

## Uncertainty and failure gates

The volume interval combines a local Jacobian linearization (using a 1.96 standard-deviation multiplier) with the envelope of near-scoring alternative fits. It is conditional on the chosen shape family, ground reference, assumed measurement errors and priors. It is not a calibrated confidence interval or proof that the hidden shape is correct; calibration is [issue #5](https://github.com/emmeyteja/autonomous-dozer/issues/5).

Statuses are `insufficient_data`, `optimization_failed`, `inconsistent_mass`, `needs_rescan`, or `estimated`. Rescan reasons include high pile-support residuals, locally unidentifiable parameters, boundary contact, slope saturation, insufficient observed free ground, too much hidden volume, broad alternative-volume intervals, and mass disagreement exceeding three supplied volume standard deviations. Research defaults include 0.08 m pile RMSE (or a higher three-sigma noise threshold), 65% hidden volume, and interval width of 50% of estimated volume; these are configurable diagnostics, not accepted machine/job limits.

A failed diagnostic may retain a point estimate for investigation; `rescan_required=true` and `volume_usable_without_rescan=false` prohibit interpreting it as an accepted estimate. Mass may constrain total volume without resolving arbitrary hidden geometry. The current interval does not model shape families outside cone/ridge, correlated scans, GNSS/pose error, nonhorizontal ground, density bias beyond the supplied prior, camera semantics or before/after surface differencing.

## Verification scope

Deterministic partial-view development fixtures use 0.25 m grids, 0.01 m independent height noise, 2.4 m footprint radius, 1.4 m peak and optional 1.2 m ridge half-length. Tests require center/radius recovery within 0.1 m, total volume within 5%, and hidden volume within 8% with an independent consistent synthetic ticket. An oblique ridge also recovers without a mass prior.

Other tests require a narrow end-cap view to request rescanning, incompatible mass to fail explicitly, separated asymmetric dumps to expose model mismatch, ground-datum shifts to preserve excess geometry, and empty/sparse scans and exhausted solver budgets to fail without confident geometry. These targeted fixtures are not a general accuracy benchmark; broader held-out sweeps and uncertainty coverage remain issue #5.
