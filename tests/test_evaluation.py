import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from autonomous_dozer.domain import Grid, Terrain
from autonomous_dozer.observations import SensorPose, observe
from autonomous_dozer.estimation import PileEstimate, FitConfig
from autonomous_dozer.evaluation import scenario, score_estimate, summarize, run_sweep

GATES = json.loads(Path('configs/estimator_promotion_gates_v1.json').read_text())


def test_partitions_are_reproducible_and_disjoint():
    a, seed_a, _, params_a = scenario('heldout', 0, 'ridge')
    b, seed_b, _, params_b = scenario('heldout', 0, 'ridge')
    c, seed_c, _, _ = scenario('development', 0, 'ridge')
    assert seed_a == seed_b != seed_c and params_a == params_b
    assert np.array_equal(a.height_m, b.height_m)
    assert not np.array_equal(a.height_m, c.height_m)


def test_failed_estimate_stays_in_denominators():
    truth = Terrain(Grid(4, 4, 1), np.ones((4, 4)))
    obs = observe(truth, SensorPose(.5, .5, 2))
    failure = PileEstimate('insufficient_data', ('no_data',), 0, 0, 'site')
    row = score_estimate(truth, obs, failure, GATES)
    row.update(supported_shape=True, split='heldout', profile='test')
    report = summarize([row], GATES)
    assert report['all']['cases'] == 1
    assert report['all']['missing_fit_rate_all_cases'] == 1
    assert report['all']['quality_success_rate_all_cases'] == 0
    assert report['all']['interval_coverage_all_cases'] == 0
    assert report['all']['metrics']['total_relative_error']['missing_cases'] == 1
    assert report['rescan_recall_on_poor_or_missing_fits'] == 1
    assert not report['promotion_gates_passed'] and not report['planning_enabled']


def test_bad_acceptance_and_nonheldout_data_do_not_mask_heldout_failure():
    truth = Terrain(Grid(4, 4, 1), np.ones((4, 4)))
    obs = observe(truth, SensorPose(.5, .5, 2))
    wrong = PileEstimate('estimated', (), 0, 0, 'site', predicted_height_m=np.zeros((4, 4)))
    row = score_estimate(truth, obs, wrong, GATES)
    row.update(supported_shape=True, split='heldout', profile='test')
    other = {**row, 'split': 'train', 'quality_pass': True}
    first = summarize([row], GATES)
    expanded = summarize([row]+[other]*200, GATES)
    assert first['promotion_checks'] == expanded['promotion_checks']
    assert expanded['heldout_accepted_bad_fraction'] == 1
    assert expanded['accepted_bad_fraction'] < .01
    assert not expanded['promotion_checks']['accepted_bad_fraction']


def test_solver_exception_recorded_without_dropping_case(monkeypatch):
    import autonomous_dozer.evaluation as module
    def failing_fit(obs, area, **kwargs):
        assert not isinstance(obs, Terrain)
        raise RuntimeError('fixture solver failure')
    monkeypatch.setattr(module, 'fit_pile', failing_fit)
    rows, summary = run_sweep(gates=GATES, splits=('heldout',), shapes=('cone',), profiles=('reference',))
    assert len(rows) == 1 and rows[0]['status'] == 'evaluation_fit_error'
    assert summary['all']['missing_fit_rate_all_cases'] == 1


def test_perfect_fit_scores_volume_shape_and_empirical_coverage():
    truth, _, _, _ = scenario('heldout', 0, 'cone')
    obs = observe(truth, SensorPose(-6, 0, 5))
    total = truth.volume_m3
    result = PileEstimate('estimated', (), 0, 0, 'site',
                          diagnostics={'conditional_volume_interval_m3': [total*.9, total*1.1],
                                       'jacobian_rank': 4, 'parameter_count': 4},
                          predicted_height_m=truth.height_m)
    row = score_estimate(truth, obs, result, GATES)
    assert row['quality_pass'] and row['interval_covers_truth']
    assert row['footprint_iou'] == 1 and row['height_rmse_m'] == 0


def test_repeated_sweep_has_same_cases_and_metrics_except_runtime():
    opts = dict(gates=GATES, splits=('train','development','heldout'), shapes=('cone',),
                profiles=('reference',), fit_config=FitConfig(starts_per_model=2))
    a, sa = run_sweep(**opts); b, sb = run_sweep(**opts)
    for row in a+b:
        row.pop('runtime_s')
    assert a == b and sa == sb
    assert len({r['scenario_seed'] for r in a}) == 3


def test_empty_sweep_is_rejected():
    with pytest.raises(ValueError):
        run_sweep(gates=GATES, splits=())
    with pytest.raises(ValueError):
        summarize([], GATES)



def test_duplicate_cases_and_oversized_sweeps_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        run_sweep(gates=GATES, profiles=('reference','reference'))
    with pytest.raises(ValueError, match="budget"):
        run_sweep(gates=GATES, max_cases=1)
