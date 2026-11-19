from dataclasses import replace
import json
import math
import numpy as np
import pytest
from autonomous_dozer.domain import Grid, Rect, Terrain
from autonomous_dozer.terrain import pile
from autonomous_dozer.observations import SensorModel, SensorPose, observe
from autonomous_dozer.observation_cli import write_observation
from autonomous_dozer.estimation import FitConfig, GroundReference, MassPrior, fit_pile
from autonomous_dozer.estimation_cli import load_observation

GRID = Grid(64, 56, 0.25, -8, -7)
AREA = Rect(-6, 6, -5, 5)


def scene(half_length=0, yaw=0.5, eye_z=2):
    truth = pile(GRID, center=(-0.3, 0.2), radius_m=2.4, height_m=1.4,
                 half_length_m=half_length, yaw_rad=yaw)
    observation = observe(truth, SensorPose(-6, 0, eye_z),
                          SensorModel(height_noise_std_m=0.01), seed=7)
    return truth, observation


@pytest.mark.parametrize("half_length", [0, 1.2])
def test_partial_fit_recovers_geometry_and_hidden_volume(half_length):
    truth, obs = scene(half_length)
    # Synthetic ticket is an independent scenario input, using continuous shape
    # volume, not a hidden-height array passed to the estimator.
    expected_volume = math.pi*2.4**2*1.4/3+2*half_length*2.4*1.4
    estimate = fit_pile(obs, AREA, mass=MassPrior(expected_volume*1500, 1500))
    assert estimate.status == "estimated", estimate.reasons
    assert estimate.selected_model == ("cone" if half_length == 0 else "ridge")
    assert estimate.parameters["cx_m"] == pytest.approx(-0.3, abs=0.1)
    assert estimate.parameters["cy_m"] == pytest.approx(0.2, abs=0.1)
    assert estimate.parameters["radius_m"] == pytest.approx(2.4, abs=0.1)
    assert estimate.diagnostics["total_volume_m3"] == pytest.approx(truth.volume_m3, rel=0.05)
    expected_hidden = float(truth.height_m[~obs.observed].sum()*GRID.area_m2)
    assert estimate.diagnostics["hidden_volume_m3"] == pytest.approx(expected_hidden, rel=0.08)
    d = estimate.diagnostics
    assert d["total_volume_m3"] == pytest.approx(d["visible_volume_m3"]+d["hidden_volume_m3"])
    assert d["volume_usable_without_rescan"]
    assert not estimate.predicted_height_m.flags.writeable
    assert estimate.source_version == obs.source_version
    assert estimate.frame_id == obs.grid.frame_id


def test_no_mass_oblique_ridge_recovers_without_truth_input():
    truth, obs = scene(1.2)
    estimate = fit_pile(obs, AREA)
    assert estimate.status == "estimated"
    assert estimate.selected_model == "ridge"
    assert estimate.diagnostics["total_volume_m3"] == pytest.approx(truth.volume_m3, rel=0.05)


def test_end_cap_can_hide_longer_ridge_and_requests_rescan():
    _, obs = scene(1.2, yaw=0.4, eye_z=3)
    x, _ = GRID.centers()
    valid = obs.observed & (x < -1.6)
    partial = replace(obs, observed=valid, height_m=np.where(valid, obs.height_m, np.nan),
                      height_std_m=np.where(valid, obs.height_std_m, np.nan))
    estimate = fit_pile(partial, AREA)
    assert estimate.rescan_required
    assert "ambiguous_volume" in estimate.reasons
    low, high = estimate.diagnostics["conditional_volume_interval_m3"]
    assert high > 1.5*low


def test_empty_or_ground_only_scan_is_not_confident_zero_volume():
    truth, obs = scene()
    blank = np.full(GRID.shape, np.nan)
    empty = replace(obs, observed=np.zeros(GRID.shape, bool), height_m=blank, height_std_m=blank)
    result = fit_pile(empty, AREA, mass=MassPrior(15000, 1500))
    assert result.status == "insufficient_data" and result.rescan_required
    assert result.predicted_height_m is None
    ground = observe(Terrain(GRID, np.zeros(GRID.shape)), SensorPose(-6, 0, 2))
    assert fit_pile(ground, AREA).status == "insufficient_data"


def test_inconsistent_mass_fails_explicitly():
    _, obs = scene(eye_z=6)
    result = fit_pile(obs, AREA, mass=MassPrior(300000, 1500, 0.05))
    assert result.status == "inconsistent_mass"
    assert "mass_prior_conflict" in result.reasons
    assert result.diagnostics["mass_residual_std"] > 3
    assert not result.diagnostics["volume_usable_without_rescan"]


def test_separated_asymmetric_dumps_expose_model_mismatch():
    a = pile(GRID, center=(-2.7, -1.8), radius_m=1.3, height_m=0.9)
    b = pile(GRID, center=(2.3, 1.8), radius_m=1.4, height_m=0.95)
    truth = Terrain(GRID, a.height_m+b.height_m)
    obs = observe(truth, SensorPose(-6, 0, 6), SensorModel(height_noise_std_m=0.01))
    result = fit_pile(obs, AREA)
    assert result.status == "needs_rescan"
    assert "shape_model_mismatch" in result.reasons
    assert result.diagnostics["pile_rmse_m"] > 0.2


def test_ground_reference_datum_shift_preserves_fit():
    _, obs = scene(1.2)
    a = fit_pile(obs, AREA)
    shifted = replace(obs, height_m=obs.height_m+10,
                      sensor_pose=replace(obs.sensor_pose, z_m=obs.sensor_pose.z_m+10))
    b = fit_pile(shifted, AREA, ground=GroundReference(10))
    assert b.diagnostics["total_volume_m3"] == pytest.approx(a.diagnostics["total_volume_m3"], rel=1e-5)


def test_solver_budget_and_seed_reproducibility():
    _, obs = scene(1.2)
    failed = fit_pile(obs, AREA, config=FitConfig(max_nfev=1))
    assert failed.status == "optimization_failed" and failed.rescan_required
    a, b = fit_pile(obs, AREA), fit_pile(obs, AREA)
    assert a.parameters == b.parameters
    assert np.array_equal(a.predicted_height_m, b.predicted_height_m)
    assert a.diagnostics == b.diagnostics


def test_observations_are_not_mutated_and_truth_type_is_rejected():
    truth, obs = scene()
    original = obs.height_m.copy()
    fit_pile(obs, AREA)
    np.testing.assert_equal(obs.height_m, original)
    with pytest.raises(TypeError, match="never"):
        fit_pile(truth, AREA)


def test_partial_scan_loader_roundtrip_and_schema_validation(tmp_path):
    _, obs = scene()
    write_observation(obs, tmp_path/"scan")
    loaded = load_observation(tmp_path/"scan")
    np.testing.assert_equal(loaded.height_m, obs.height_m)
    assert loaded.sensor_pose == obs.sensor_pose
    assert loaded.source_version == obs.source_version
    p = tmp_path/"scan"/"observation.json"
    data = json.loads(p.read_text()); data["schema"] = "unknown"; p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        load_observation(tmp_path/"scan")


@pytest.mark.parametrize("constructor,kwargs", [
    (GroundReference, {"std_m": -1}), (GroundReference, {"elevation_m": math.nan}),
    (MassPrior, {"mass_kg": 0, "bulk_density_kg_m3": 1500}),
    (MassPrior, {"mass_kg": 10, "bulk_density_kg_m3": -1}),
    (MassPrior, {"mass_kg": 10, "bulk_density_kg_m3": 1500, "relative_volume_std": 1}),
    (FitConfig, {"starts_per_model": 1}), (FitConfig, {"max_nfev": 0}),
    (FitConfig, {"max_samples": 1}), (FitConfig, {"seed": -1}),
    (FitConfig, {"max_hidden_fraction": 1.1}), (FitConfig, {"noise_floor_m": 0}),
])
def test_invalid_fit_contracts(constructor, kwargs):
    with pytest.raises(ValueError):
        constructor(**kwargs)


def test_invalid_area_and_resource_budget():
    _, obs = scene()
    with pytest.raises(ValueError, match="inside"):
        fit_pile(obs, Rect(-9, 9, -8, 8))
    with pytest.raises(ValueError, match="budget"):
        fit_pile(obs, AREA, config=FitConfig(max_grid_cells=10))
    with pytest.raises(ValueError, match="small"):
        fit_pile(obs, Rect(0, 0.1, 0, 0.1))


def test_cli_writes_explicit_insufficient_data_report(tmp_path, monkeypatch):
    import sys
    from autonomous_dozer.estimation_cli import main
    _, obs = scene()
    empty = replace(obs, observed=np.zeros(GRID.shape, bool),
                    height_m=np.full(GRID.shape, np.nan), height_std_m=np.full(GRID.shape, np.nan))
    write_observation(empty, tmp_path/"input")
    monkeypatch.setattr(sys, "argv", ["estimate", "--observation", str(tmp_path/"input"),
                                     "--output", str(tmp_path/"output")])
    assert main() == 2
    report = json.loads((tmp_path/"output"/"estimate.json").read_text())
    assert report["status"] == "insufficient_data"
    assert report["rescan_required"] and not report["planner_executed"]
    assert not (tmp_path/"output"/"estimate.npz").exists()



def test_common_ground_uncertainty_is_not_averaged_away():
    _, obs = scene(1.2)
    result = fit_pile(obs, AREA, ground=GroundReference(std_m=0.04))
    floor = result.diagnostics["common_ground_volume_std_floor_m3"]
    assert floor > 0.5
    assert result.diagnostics["conditional_local_volume_std_m3"] >= floor
