from dataclasses import replace
import numpy as np
import pytest
from autonomous_dozer.domain import Blade, Grid, Rect, Terrain
from autonomous_dozer.transport import CarryState, HeightmapTransport, Stroke


def scene():
    g = Grid(20, 8, 0.5)
    cut, release = Rect(1, 3, 1, 3), Rect(6, 9, 1, 3)
    h = np.zeros(g.shape)
    h[g.mask(cut)] = 0.5
    t = Terrain(g, h)
    b = Blade(2, 1, 2, capacity_factor=1)
    s = Stroke(0, ((0.5, 2), (9.5, 2)), cut, release,
               max_cut_depth_m=0.5, max_deposit_height_m=0.5)
    return t, b, s


def run(t, b, s, carry=CarryState()):
    return HeightmapTransport().apply(t, s, b, carry)


def test_sparse_path_carry_release_and_slumping_conserve_mass():
    t, b, s = scene()
    result = run(t, b, s)
    assert len(result.steps) == 18
    assert sum(p.cut_m3 for p in result.steps) == pytest.approx(2)
    assert sum(p.released_m3 for p in result.steps) == pytest.approx(2)
    assert all(p.released_m3 == 0 for p in result.steps if p.x_m < 6)
    assert max(p.carried_m3 for p in result.steps) <= b.effective_capacity_m3
    assert all(abs(p.mass_error_m3) < 1e-12 for p in result.steps)
    assert abs(result.mass_error_m3) < 1e-12
    assert result.relaxation.converged and not result.calibrated
    assert t.volume_m3 == 2 and t.version == 0


def test_capacity_and_cut_depth_limits_leave_work():
    t, b, s = scene()
    limited = run(t, replace(b, rated_capacity_m3=0.5), s)
    assert 'capacity_limited' in limited.flags
    assert sum(p.cut_m3 for p in limited.steps) == pytest.approx(0.5)
    shallow = run(t, b, replace(s, max_cut_depth_m=0.1))
    assert sum(p.cut_m3 for p in shallow.steps) == pytest.approx(0.4)
    assert 'depth_limited' in shallow.flags
    assert not shallow.objective_complete


def test_full_destination_blocks_excavation():
    t, b, s = scene()
    h = t.height_m.copy()
    h[t.grid.mask(s.release)] = 0.5
    result = run(Terrain(t.grid, h), b, s)
    assert 'blocked_disposal' in result.flags
    assert sum(p.cut_m3 for p in result.steps) == 0
    assert abs(result.mass_error_m3) < 1e-12


def test_insufficient_disposal_retains_initial_load():
    t, b, s = scene()
    result = run(t, b, replace(s, max_deposit_height_m=0.1), CarryState(1))
    assert 'disposal_full' in result.flags
    assert result.carry.volume_m3 == pytest.approx(0.4)
    assert sum(p.cut_m3 for p in result.steps) == 0
    assert abs(result.mass_error_m3) < 1e-12


def test_shedding_stays_on_map_and_is_counted_as_spill():
    t, b, s = scene()
    result = run(t, b, replace(s, shedding_fraction_per_m=0.3))
    assert sum(p.shed_m3 for p in result.steps) > 0
    assert result.spill_m3 > 0
    assert abs(result.mass_error_m3) < 1e-12


def test_boundary_without_shoulders_retains_shed_material():
    t, b, s = scene()
    result = run(t, replace(b, width_m=4), replace(s, shedding_fraction_per_m=1))
    assert 'shedding_blocked' in result.flags
    assert sum(p.shed_m3 for p in result.steps) == 0
    assert abs(result.mass_error_m3) < 1e-12


def test_empty_source_and_fill_deficit():
    t, b, s = scene()
    empty = Terrain(t.grid, np.zeros(t.grid.shape))
    result = run(empty, b, s)
    assert result.objective_complete and 'no_work' in result.flags
    fill = run(empty, b, replace(s, objective='fill', target_height_m=0.1))
    assert not fill.objective_complete and 'grade_incomplete' in fill.flags
    assert fill.fill_deficit_m3 == pytest.approx(0.6)


def test_fill_reserves_only_target_capacity_and_finish_detects_overfill():
    t, b, s = scene()
    fill = run(t, b, replace(s, objective='fill', target_height_m=0.1))
    assert sum(p.cut_m3 for p in fill.steps) == pytest.approx(0.6)
    assert fill.objective_complete
    h = np.full(t.grid.shape, 0.2)
    t = Terrain(t.grid, h)
    assert run(t, b, replace(s, objective='fill', target_height_m=0.1)).objective_complete
    finish = run(t, b, replace(s, objective='finish', target_height_m=0.1))
    assert not finish.objective_complete and finish.overfill_m3 > 0


@pytest.mark.parametrize('change', [
    {'source_version': 1}, {'path_xy_m': ((0, 0.5), (9, 0.5))},
    {'path_xy_m': ((0, 2), (11, 2))}, {'max_cut_depth_m': 2},
    {'release': Rect(2, 4, 1, 3)},
])
def test_invalid_strokes_rejected(change):
    t, b, s = scene()
    with pytest.raises(ValueError):
        run(t, b, replace(s, **change))


def test_overload_coarse_resolution_and_push_limit():
    t, b, s = scene()
    with pytest.raises(ValueError, match='capacity'):
        run(t, b, s, CarryState(3))
    with pytest.raises(ValueError, match='full cut or release'):
        run(t, replace(b, width_m=0.1), s)
    with pytest.raises(ValueError, match='distance'):
        run(t, replace(b, max_push_m=2), s)


@pytest.mark.parametrize('change', [
    {'path_xy_m': ((1, 2), (0, 2))}, {'path_xy_m': ((0, 2), (1, 3))},
    {'max_cut_depth_m': float('nan')}, {'shedding_fraction_per_m': -1},
    {'objective': 'unknown'}, {'target_height_m': 2, 'objective': 'fill'},
])
def test_contract_validation(change):
    with pytest.raises(ValueError):
        replace(scene()[2], **change)


def test_partial_lane_does_not_claim_whole_release_grade_complete():
    t, b, s = scene()
    result = run(t, replace(b, width_m=1), replace(s, objective='fill', target_height_m=0.1))
    assert not result.objective_complete
    assert result.fill_deficit_m3 >= 0.3-1e-10


def test_nonconverged_slumping_is_explicit():
    t, b, s = scene()
    result = HeightmapTransport(repose_angle_deg=1, max_relax_sweeps=1).apply(t, s, b)
    assert 'relaxation_incomplete' in result.flags
    assert not result.objective_complete
    assert abs(result.mass_error_m3) < 1e-12


def test_transport_cli_report_and_no_overwrite(tmp_path):
    import json
    from autonomous_dozer.transport_cli import main
    output = tmp_path/'stroke'
    assert main(['--output', str(output)]) == 0
    report = json.loads((output/'transport.json').read_text())
    assert report['objective_complete'] and not report['calibrated']
    assert report['cut_m3'] == pytest.approx(0.6)
    assert abs(report['mass_error_m3']) < 1e-10
    assert (output/'transport.npz').exists()
    with pytest.raises(FileExistsError):
        main(['--output', str(output)])


def test_partial_cut_lane_does_not_claim_whole_source_moved():
    t, b, s = scene()
    result = run(t, replace(b, width_m=1), s)
    assert result.remaining_cut_m3 > 0
    assert not result.objective_complete


def test_post_slump_limits_are_reported_without_clipping_mass():
    t, b, s = scene()
    h = t.height_m.copy()
    h[2:6, 11] = 2  # A tall adjacent berm can slump into the destination.
    result = run(Terrain(t.grid, h), b,
                 replace(s, objective='finish', target_height_m=0.1,
                         max_deposit_height_m=0.1))
    assert 'post_slump_deposit_limit_exceeded' in result.flags
    assert not result.objective_complete
    assert abs(result.mass_error_m3) < 1e-10


def test_varied_capacity_load_and_shedding_preserve_mass():
    t, b, s = scene()
    rng = np.random.default_rng(6)
    for _ in range(12):
        capacity = float(rng.uniform(0.2, 3))
        initial = CarryState(float(rng.uniform(0, capacity)))
        result = run(t, replace(b, rated_capacity_m3=capacity),
                     replace(s, shedding_fraction_per_m=float(rng.uniform(0, 1))), initial)
        assert max(step.carried_m3 for step in result.steps) <= capacity+1e-10
        assert min(result.terrain.height_m.flat) >= 0
        assert abs(result.mass_error_m3) < 1e-10
        assert all(abs(step.mass_error_m3) < 1e-10 for step in result.steps)
