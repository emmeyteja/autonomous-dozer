"""Seeded scenario sweeps: truth remains in generation and scoring only."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
from .domain import Grid, Rect, Terrain
from .terrain import pile
from .observations import SensorPose, SensorModel, observe
from .estimation import FitConfig, MassPrior, PileEstimate, fit_pile

SPLITS = {"train": 0, "development": 1, "heldout": 2}
SHAPES = ("cone", "ridge", "asymmetric", "multi")
PROFILES = {
    "reference": {},
    "end_on": {"eye_z": 2},
    "low_view": {"eye_z": 1},
    "oblique": {"eye_z": 2, "view_angle": 0.7},
    "short_range": {"max_range": 6},
    "narrow_fov": {"fov_deg": 30},
    "noise": {"noise": 0.06},
    "dropout": {"dropout": 0.5},
    "density_low": {"density_bias": -0.15},
    "density_high": {"density_bias": 0.15},
    "without_mass": {"use_mass": False},
    "occluded_density_low": {"eye_z": 1, "density_bias": -0.15},
    "occluded_density_high": {"eye_z": 1, "density_bias": 0.15},
}


def scenario(split, index, shape, root_seed=2026):
    """Separate geometry streams by split/index/shape; profiles share a scene."""
    if split not in SPLITS or shape not in SHAPES or type(index) is not int or index < 0:
        raise ValueError("Unknown split/shape or invalid scenario index")
    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("Root seed must be a nonnegative integer")
    seed = int(np.random.SeedSequence([root_seed, SPLITS[split], index, SHAPES.index(shape)]).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    radius, slope = float(rng.uniform(1.8, 2.5)), math.tan(math.radians(float(rng.uniform(28, 36))))
    center = tuple(float(v) for v in rng.uniform(-0.35, 0.35, 2))
    half_length = float(rng.uniform(0.6, 1.2)) if shape == "ridge" else 0.0
    grid = Grid(48, 40, 0.4, -9.6, -8)
    t = pile(grid, center=center, radius_m=radius, height_m=slope*radius, half_length_m=half_length)
    ticket_volume = slope*(math.pi*radius**3/3+2*half_length*radius**2)
    if shape == "asymmetric":
        x, _ = grid.centers()
        # Unsupported asymmetric stress fixture, not a physically settled pile.
        t = Terrain(grid, t.height_m*np.where(x < center[0], 0.5, 1.5))
    elif shape == "multi":
        a = pile(grid, center=(-2.5, -1.5), radius_m=1.2, height_m=0.8)
        b = pile(grid, center=(2.5, 1.5), radius_m=1.3, height_m=0.9)
        t = Terrain(grid, a.height_m+b.height_m)
        ticket_volume = math.pi*(1.2**2*0.8+1.3**2*0.9)/3
    return t, seed, ticket_volume, {"radius_m": radius, "slope": slope, "center": center,
                                   "half_length_m": half_length, "shape": shape}


def score_estimate(truth, observation, estimate: PileEstimate, gates):
    """Evaluation only: never pass these arrays or metrics back to the fit."""
    actual = truth.height_m
    observed = observation.observed
    area = truth.grid.area_m2
    total = float(actual.sum()*area)
    visible = float(actual[observed].sum()*area)
    hidden = total-visible
    mask = actual > gates["footprint_height_threshold_m"]
    row = {"relative_error_denominator_floor_m3": area*0.01,
           "truth_total_m3": total, "truth_visible_m3": visible, "truth_hidden_m3": hidden,
           "observed_cell_fraction": float(observed.mean()),
           "observed_material_fraction": visible/total if total else None,
           "status": estimate.status, "rescan_required": estimate.rescan_required,
           "fit_available": estimate.predicted_height_m is not None,
           "total_relative_error": None, "visible_relative_error": None, "hidden_relative_error": None,
           "height_rmse_m": None, "footprint_iou": None, "interval_available": False,
           "interval_covers_truth": False, "quality_pass": False}
    pred = estimate.predicted_height_m
    if pred is not None:
        inferred_mask = pred > gates["footprint_height_threshold_m"]
        union = mask | inferred_mask
        row.update(
            total_relative_error=abs(float(pred.sum()*area)-total)/max(total, area*0.01),
            visible_relative_error=abs(float(pred[observed].sum()*area)-visible)/max(visible, area*0.01),
            hidden_relative_error=abs(float(pred[~observed].sum()*area)-hidden)/max(hidden, area*0.01),
            height_rmse_m=float(np.sqrt(np.mean((pred[union]-actual[union])**2))) if union.any() else 0,
            footprint_iou=float(np.count_nonzero(mask & inferred_mask)/np.count_nonzero(union)) if union.any() else 1)
        row["quality_pass"] = bool(row["total_relative_error"] <= gates["max_total_relative_error"]
                                   and row["hidden_relative_error"] <= gates["max_hidden_relative_error"]
                                   and row["height_rmse_m"] <= gates["max_height_rmse_m"]
                                   and row["footprint_iou"] >= gates["min_footprint_iou"])
        interval = estimate.diagnostics.get("conditional_volume_interval_m3")
        # Unidentifiable fits have no trustworthy local interval even if the
        # finite multistart envelope happens to be narrow.
        if interval is not None and estimate.diagnostics.get("jacobian_rank") == estimate.diagnostics.get("parameter_count"):
            row["interval_available"] = True
            row["interval_covers_truth"] = bool(interval[0] <= total <= interval[1])
    return row


def summarize(rows, gates):
    """All failures stay in rate denominators; means name their available n."""
    if not rows:
        raise ValueError("Cannot summarize an empty sweep")
    supported = [r for r in rows if r["supported_shape"]]
    heldout = [r for r in supported if r["split"] == "heldout"]
    accepted = [r for r in rows if not r["rescan_required"]]
    heldout_accepted = [r for r in rows if r["split"] == "heldout" and not r["rescan_required"]]
    bad = [r for r in rows if not r["quality_pass"]]
    def rate(items, predicate):
        return sum(bool(predicate(r)) for r in items)/len(items) if items else None
    def group(items):
        means = {}
        for key in ("total_relative_error", "visible_relative_error", "hidden_relative_error", "height_rmse_m", "footprint_iou"):
            values = [r[key] for r in items if r[key] is not None]
            means[key] = {"available_cases": len(values), "missing_cases": len(items)-len(values),
                          "mean_available": float(np.mean(values)) if values else None}
        return {"cases": len(items), "quality_success_rate_all_cases": rate(items, lambda r: r["quality_pass"]),
                "rescan_rate_all_cases": rate(items, lambda r: r["rescan_required"]),
                "missing_fit_rate_all_cases": rate(items, lambda r: not r["fit_available"]),
                "interval_coverage_all_cases": rate(items, lambda r: r["interval_covers_truth"]),
                "interval_available_cases": sum(r["interval_available"] for r in items),
                "empirical_coverage_when_interval_available": rate([r for r in items if r["interval_available"]], lambda r: r["interval_covers_truth"]),
                "metrics": means}
    checks = {
        "enough_heldout_supported_cases": len(heldout) >= gates["min_heldout_supported_cases"],
        "heldout_quality": bool(heldout) and rate(heldout, lambda r: r["quality_pass"]) >= gates["min_supported_quality_success_rate"],
        "heldout_interval_coverage": bool(heldout) and rate(heldout, lambda r: r["interval_covers_truth"]) >= gates["min_supported_interval_coverage"],
        "heldout_rescan_rate": bool(heldout) and rate(heldout, lambda r: r["rescan_required"]) <= gates["max_supported_rescan_rate"],
        "accepted_bad_fraction": bool(heldout_accepted) and rate(heldout_accepted, lambda r: not r["quality_pass"]) <= gates["max_accepted_bad_fraction"],
    }
    return {"all": group(rows), "supported_shapes": group(supported), "heldout_supported": group(heldout),
            "accepted_cases": len(accepted), "accepted_bad_fraction": rate(accepted, lambda r: not r["quality_pass"]),
            "heldout_accepted_cases": len(heldout_accepted),
            "heldout_accepted_bad_fraction": rate(heldout_accepted, lambda r: not r["quality_pass"]),
            "poor_or_missing_fit_cases": len(bad), "rescan_recall_on_poor_or_missing_fits": rate(bad, lambda r: r["rescan_required"]),
            "by_profile": {name: group([r for r in rows if r["profile"] == name]) for name in sorted({r["profile"] for r in rows})},
            "by_split": {name: group([r for r in rows if r["split"] == name]) for name in sorted({r["split"] for r in rows})},
            "promotion_checks": checks, "promotion_gates_passed": all(checks.values()),
            "planning_enabled": False, "coverage_is_empirical_not_calibrated_confidence": True}


def run_sweep(*, gates, splits=("train", "development", "heldout"), shapes=SHAPES,
              profiles=tuple(PROFILES), scenes_per_split=1, root_seed=2026, fit_config=FitConfig(), max_cases=10000):
    if type(scenes_per_split) is not int or scenes_per_split < 1:
        raise ValueError("scenes_per_split must be positive")
    if (not splits or not shapes or not profiles or any(len(set(values)) != len(values)
                                                        for values in (splits, shapes, profiles))):
        raise ValueError("Require nonempty selections without duplicate cases")
    if type(max_cases) is not int or max_cases < 1 or len(splits)*len(shapes)*len(profiles)*scenes_per_split > max_cases:
        raise ValueError("Sweep case budget exceeded")
    if any(p not in PROFILES for p in profiles):
        raise ValueError("Unknown sensor profile")
    rows = []
    for split in splits:
        for index in range(scenes_per_split):
            for shape in shapes:
                truth, seed, ticket_volume, parameters = scenario(split, index, shape, root_seed)
                for name in profiles:
                    p = dict(eye_z=5.0, view_angle=0.0, max_range=25.0, fov_deg=360.0,
                             noise=0.01, dropout=0.0, density_bias=0.0, use_mass=True)
                    p.update(PROFILES[name])
                    pose = SensorPose(-6*math.cos(p["view_angle"]), -6*math.sin(p["view_angle"]), p["eye_z"], p["view_angle"])
                    obs = observe(truth, pose, SensorModel(max_range_m=p["max_range"], horizontal_fov_rad=math.radians(p["fov_deg"]),
                                                          height_noise_std_m=p["noise"], dropout_probability=p["dropout"]), seed=seed)
                    prior = MassPrior(ticket_volume*1500, 1500*(1+p["density_bias"])) if p["use_mass"] else None
                    start = time.perf_counter()
                    try:
                        estimate = fit_pile(obs, Rect(-6, 6, -5, 5), mass=prior, config=fit_config)
                    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                        estimate = PileEstimate("evaluation_fit_error", (type(exc).__name__+": "+str(exc),),
                                                obs.source_version, obs.stamp_s, obs.grid.frame_id)
                    elapsed = time.perf_counter()-start
                    row = score_estimate(truth, obs, estimate, gates)
                    row.update(case_id=f"{split}-{index:03d}-{shape}-{name}", split=split, shape=shape,
                               supported_shape=shape in {"cone", "ridge"}, profile=name, scenario_seed=seed,
                               parameters=parameters, grid=asdict(truth.grid), sensor_pose=asdict(pose),
                               mass_prior=asdict(prior) if prior else None, sensor_profile=p,
                               selected_model=estimate.selected_model,
                               volume_interval_m3=estimate.diagnostics.get("conditional_volume_interval_m3"),
                               runtime_s=elapsed, reasons=estimate.reasons)
                    rows.append(row)
    return rows, summarize(rows, gates)
