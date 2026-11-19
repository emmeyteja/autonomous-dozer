import math
import numpy as np
import pytest
from autonomous_dozer.domain import Blade, Grid, Job, Rect, Terrain, WorkPass
from autonomous_dozer.terrain import pile, multi_dump
from autonomous_dozer.simulation import apply_pass, relax
from autonomous_dozer.planning import InfeasibleJob, next_pass
from autonomous_dozer.cli import run_demo


@pytest.fixture
def blade():
    return Blade(3.25, 1.195, 3.4)


def test_terrain_owns_immutable_snapshot():
    h = np.ones((4, 4))
    t = Terrain(Grid(4, 4, 1), h)
    h[:] = 0
    assert t.volume_m3 == 16
    with pytest.raises(ValueError):
        t.height_m[0, 0] = 3


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1])
def test_invalid_terrain_rejected(invalid):
    with pytest.raises(ValueError):
        Terrain(Grid(2, 2, 1), np.full((2, 2), invalid))


def test_cone_volume_matches_continuous_geometry():
    t = pile(Grid(200, 200, 0.05, -5, -5), radius_m=3, height_m=2)
    expected = math.pi*3**2*2/3
    assert t.volume_m3 == pytest.approx(expected, rel=0.001)


def test_multidump_reproducible():
    g = Grid(40, 40, 0.25, -5, -5)
    assert np.array_equal(multi_dump(g, 8).height_m, multi_dump(g, 8).height_m)
    assert not np.array_equal(multi_dump(g, 8).height_m, multi_dump(g, 9).height_m)


def test_relaxation_conserves_material_and_reaches_cardinal_repose():
    h = np.zeros((20, 20)); h[10, 10] = 9
    t = Terrain(Grid(20, 20, 0.5), h)
    result = relax(t)
    assert result.converged
    assert result.terrain.volume_m3 == pytest.approx(t.volume_m3, abs=1e-10)
    assert result.terrain.height_m.min() >= 0
    for axis in (0, 1):
        assert np.max(np.abs(np.diff(result.terrain.height_m, axis=axis))) <= math.tan(math.radians(32))*0.5+1e-5


def test_unconverged_relaxation_is_reported():
    h = np.zeros((20, 20)); h[10, 10] = 100
    assert not relax(Terrain(Grid(20, 20, 1), h), max_sweeps=1).converged


def test_transfer_capacity_floor_and_outside_cells():
    g = Grid(10, 4, 1)
    h = np.zeros(g.shape); h[:, :4] = 1
    t = Terrain(g, h)
    p = WorkPass(0, Rect(0, 4, 0, 4), Rect(7, 10, 0, 4), 0.2, 1.7)
    out = apply_pass(t, p)
    assert out.moved_m3 == pytest.approx(1.7)
    assert out.terrain.volume_m3 == pytest.approx(t.volume_m3)
    assert out.terrain.height_m[:, :4].min() >= 0.2
    assert np.array_equal(out.terrain.height_m[:, 4:7], t.height_m[:, 4:7])
    assert np.array_equal(t.height_m, h)
    with pytest.raises(ValueError, match="Stale"):
        apply_pass(out.terrain, p)


def test_invalid_transfer_rejected():
    t = Terrain(Grid(10, 4, 1), np.ones((4, 10)))
    for dest in [Rect(1, 6, 0, 4), Rect(9, 12, 0, 4), Rect(0, 1, 0, 4)]:
        with pytest.raises(ValueError):
            apply_pass(t, WorkPass(0, Rect(2, 5, 0, 4), dest, 0, 1))


def test_no_work_and_infeasible_job(blade):
    g = Grid(48, 40, 0.5, -12, -10)
    t = Terrain(g, np.zeros(g.shape))
    job = Job(Rect(-6, 4, -5, 5), Rect(7, 11, -4, 4), 0)
    assert next_pass(t, job, blade) is None
    with pytest.raises(InfeasibleJob):
        next_pass(t, job, Blade(3, 1, 3, max_push_m=2))
    with pytest.raises(ValueError):
        next_pass(t, job, blade, overlap=float("nan"))


def test_lane_width_layer_and_version(blade):
    g = Grid(48, 40, 0.5, -12, -10)
    t = pile(g, center=(-1, 0))
    p = next_pass(t, Job(Rect(-6, 4, -5, 5), Rect(7, 11, -4, 4), 0), blade)
    assert p.cut.ymax_m-p.cut.ymin_m <= blade.width_m
    assert float(t.height_m.max())-p.cut_height_m <= 0.75*blade.face_height_m+1e-12
    assert p.source_version == t.version


@pytest.mark.parametrize("shape", ["cone", "ridge", "multi"])
def test_end_to_end_conserves_material_and_removes_upper_excess(blade, shape):
    initial, final, report = run_demo(blade, pile_type=shape, resolution_m=0.5)
    assert report["status"] == "upper_excess_within_tolerance"
    assert final.volume_m3 == pytest.approx(initial.volume_m3, abs=1e-9)
    assert report["final"]["max_excess_m"] <= 0.03
    assert all(h["moved_m3"] <= blade.effective_capacity_m3+1e-9 for h in report["history"])
    assert all(h["relaxation_converged"] for h in report["history"])


def test_pass_limit_is_not_success(blade):
    _, _, report = run_demo(blade, max_passes=1)
    assert report["status"] == "pass_limit"


def test_upper_excess_status_does_not_claim_holes_are_filled(blade):
    _, _, report = run_demo(blade, target_height_m=0.4)
    assert report["status"] == "upper_excess_within_tolerance"
    assert report["final"]["grade_rmse_m"] > 0.03
    assert report["final"]["fraction_within_grade_tolerance"] < 1


def test_coarse_grid_and_nonfinite_blade_rejected():
    with pytest.raises(ValueError):
        Blade(float("nan"), 1, 2)
    g = Grid(6, 6, 4, -12, -12)
    t = Terrain(g, np.ones(g.shape))
    with pytest.raises(InfeasibleJob, match="coarse"):
        next_pass(t, Job(Rect(-8, 0, -8, 0), Rect(4, 12, -8, 0), 0), Blade(3, 1, 2))
