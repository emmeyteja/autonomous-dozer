from dataclasses import replace
import math
import json
import numpy as np
import pytest
from autonomous_dozer.domain import Grid, Terrain, Blade, Job, Rect
from autonomous_dozer.observations import SensorPose, SensorModel, Observation, observe
from autonomous_dozer.observation_cli import write_observation
from autonomous_dozer.planning import next_pass


def flat(nx=10, ny=5):
    return Terrain(Grid(nx, ny, 1), np.zeros((ny, nx)), version=4)


def noiseless(**kwargs):
    return SensorModel(height_noise_std_m=0, **kwargs)


def test_flat_ground_has_valid_zero_heights():
    t = flat()
    o = observe(t, SensorPose(0.5, 2.5, 1), noiseless(), stamp_s=3.25)
    assert o.observed.all()
    assert np.all(o.height_m == 0)
    assert np.all(o.height_std_m == 0)
    assert o.source_version == 4 and o.stamp_s == 3.25
    assert o.grid.frame_id == "site"


def test_wall_occludes_hidden_side_and_other_view_sees_it():
    h = np.zeros((5, 10)); h[:, 4] = 2
    t = Terrain(Grid(10, 5, 1), h)
    front = observe(t, SensorPose(0.5, 2.5, 1), noiseless())
    rear = observe(t, SensorPose(9.5, 2.5, 1, math.pi), noiseless())
    assert front.observed[2, 2] and not front.observed[2, 7]
    assert np.isnan(front.height_m[2, 7])
    assert rear.observed[2, 7] and not rear.observed[2, 2]


def test_diagonal_ray_wall_and_clear_corridor():
    h = np.zeros((7, 7)); h[3, 3] = 2
    o = observe(Terrain(Grid(7, 7, 1), h), SensorPose(0.5, 0.5, 1), noiseless())
    assert not o.observed[6, 6]
    assert o.observed[6, 1]


def test_top_above_eye_not_visible_through_its_own_cell_face():
    h = np.zeros((5, 10)); h[:, 4] = 2
    o = observe(Terrain(Grid(10, 5, 1), h), SensorPose(0.5, 2.5, 1), noiseless())
    assert not o.observed[2, 4]
    assert np.isnan(o.height_m[2, 4])


def test_reflection_invariance():
    h = np.zeros((7, 11)); h[1:6, 4] = 1.2; h[3, 7] = 0.7
    g = Grid(11, 7, 1)
    a = observe(Terrain(g, h), SensorPose(0.5, 3.5, 1.5), noiseless())
    b = observe(Terrain(g, h[:, ::-1]), SensorPose(10.5, 3.5, 1.5), noiseless())
    assert np.array_equal(a.observed, b.observed[:, ::-1])


def test_fov_yaw_and_slant_range():
    t = flat()
    east = observe(t, SensorPose(4.5, 2.5, 1), noiseless(horizontal_fov_rad=math.pi/2, max_range_m=3))
    west = observe(t, SensorPose(4.5, 2.5, 1, math.pi), noiseless(horizontal_fov_rad=math.pi/2, max_range_m=3))
    assert east.observed[2, 6] and not east.observed[2, 2]
    assert west.observed[2, 2] and not west.observed[2, 6]
    # Horizontal distance=3, slant=sqrt(10)>3.
    assert not east.observed[2, 7]
    near = observe(t, SensorPose(4.5, 2.5, 1), noiseless(min_range_m=2))
    assert not near.observed[2, 4]


def test_seeded_noise_and_dropout_are_reproducible_and_independent():
    t, p = flat(30, 20), SensorPose(0.5, 10.5, 1)
    m = SensorModel(max_range_m=100, height_noise_std_m=0.04, dropout_probability=0.3)
    a, b = observe(t, p, m, seed=123), observe(t, p, m, seed=123)
    assert np.array_equal(a.observed, b.observed)
    np.testing.assert_equal(a.height_m, b.height_m)
    c = observe(t, p, replace(m, dropout_probability=0), seed=123)
    assert np.array_equal(a.height_m[a.observed], c.height_m[a.observed])
    assert a.observed.sum() < c.observed.sum()
    d = observe(t, p, m, seed=124)
    assert not np.array_equal(a.observed, d.observed)


def test_noise_is_unbiased_and_negative_ground_measurements_are_not_clipped():
    o = observe(flat(64, 64), SensorPose(0.5, 32.5, 1),
                SensorModel(max_range_m=100, height_noise_std_m=0.04), seed=91)
    assert o.observed.all()
    assert abs(float(o.height_m.mean())) < 0.003
    assert 0.037 < float(o.height_m.std()) < 0.043
    assert np.any(o.height_m < 0)
    assert np.all(o.height_std_m == 0.04)


def test_total_dropout_is_unknown_and_not_zero_ground():
    o = observe(flat(), SensorPose(0.5, 2.5, 1), noiseless(dropout_probability=1))
    assert not o.observed.any()
    assert np.isnan(o.height_m).all() and np.isnan(o.height_std_m).all()
    with pytest.raises(ValueError, match="Unobserved"):
        replace(o, height_m=np.zeros(o.grid.shape))


def test_observation_immutability_and_no_truth_payload():
    t = flat()
    original = t.height_m.copy()
    o = observe(t, SensorPose(0.5, 2.5, 1))
    assert np.array_equal(t.height_m, original)
    assert not any("truth" in key for key in o.__dict__)
    for a in (o.height_m, o.observed, o.height_std_m):
        assert not a.flags.writeable
    supplied = o.height_m.copy()
    other = replace(o, height_m=supplied)
    supplied[:] = 100
    assert not np.any(other.height_m == 100)


@pytest.mark.parametrize("kwargs", [
    {"height_noise_std_m": -1}, {"height_noise_std_m": math.nan},
    {"max_range_m": math.inf}, {"min_range_m": 30},
    {"horizontal_fov_rad": 0}, {"horizontal_fov_rad": 7},
    {"dropout_probability": -0.1}, {"dropout_probability": 1.1},
    {"max_ray_cell_visits": 0},
])
def test_invalid_sensor_models_rejected(kwargs):
    with pytest.raises(ValueError):
        SensorModel(**kwargs)


@pytest.mark.parametrize("pose", [SensorPose(-1, 2, 1), SensorPose(10, 2, 1), SensorPose(1, 2, 0)])
def test_invalid_sensor_placement_rejected(pose):
    with pytest.raises(ValueError):
        observe(flat(), pose)


def test_invalid_metadata_and_budget_rejected():
    t, p = flat(), SensorPose(0.5, 2.5, 1)
    with pytest.raises(ValueError, match="budget"):
        observe(t, p, noiseless(max_ray_cell_visits=1))
    for seed in (-1, True, 1.5):
        with pytest.raises(ValueError, match="Seed"):
            observe(t, p, seed=seed)
    with pytest.raises(ValueError):
        observe(t, p, stamp_s=math.nan)
    with pytest.raises(ValueError):
        SensorPose(math.inf, 0, 1)
    o = observe(t, p)
    with pytest.raises(ValueError):
        replace(o, observed=o.observed.astype(float))
    with pytest.raises(ValueError):
        replace(o, height_m=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        replace(o, source_version=-1)


def test_partial_observation_cannot_enter_oracle_planner():
    o = observe(flat(), SensorPose(0.5, 2.5, 1))
    with pytest.raises(TypeError, match="partial"):
        next_pass(o, Job(Rect(0, 4, 0, 4), Rect(6, 10, 0, 4), 0), Blade(3, 1, 3))


def test_measurement_export_no_truth_and_no_overwrite(tmp_path):
    o = observe(flat(), SensorPose(0.5, 2.5, 1), noiseless(dropout_probability=1))
    path = tmp_path/"scan"
    report = write_observation(o, path)
    with np.load(path/"observation.npz", allow_pickle=False) as data:
        assert set(data.files) == {"height_m", "observed", "height_std_m"}
        assert np.isnan(data["height_m"]).all()
    assert report["unknown_cells"] == 50 and report["observed_cells"] == 0
    assert not report["planner_executed"]
    assert json.loads((path/"observation.json").read_text())["source_version"] == 4
    with pytest.raises(FileExistsError):
        write_observation(o, path)



def test_nadir_not_excluded_by_yaw_and_edge_sensor_stays_inside_grid():
    o = observe(flat(), SensorPose(4.5, 2.5, 1, math.pi),
                noiseless(horizontal_fov_rad=0.1))
    assert o.observed[2, 4]
    edge = observe(flat(), SensorPose(0, 0, 1), noiseless())
    assert edge.observed.all()


def test_source_column_can_occlude_low_neighbor():
    h = np.zeros((5, 10)); h[2, 0] = 2
    o = observe(Terrain(Grid(10, 5, 1), h), SensorPose(0.5, 2.5, 2.1), noiseless(min_range_m=0))
    assert o.observed[2, 0]
    assert not o.observed[2, 1]
