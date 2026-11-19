"""A single uncalibrated blade stroke; ground-truth simulation inputs only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .domain import Blade, Grid, Rect, Terrain
from .transport import HeightmapTransport, Stroke


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--objective', choices=('move', 'fill', 'finish'), default='fill')
    parser.add_argument('--shedding-fraction-per-m', type=float, default=0.0)
    args = parser.parse_args(argv)
    grid = Grid(40, 16, 0.25)
    cut, release = Rect(1, 3, 1, 3), Rect(6, 9, 1, 3)
    h = np.zeros(grid.shape)
    h[grid.mask(cut)] = 0.8
    terrain = Terrain(grid, h)
    # Synthetic fixture, not an OEM blade or a calibrated machine profile.
    blade = Blade(2, 1, 2, capacity_factor=1)
    stroke = Stroke(0, ((0.5, 2), (9.5, 2)), cut, release,
                    max_cut_depth_m=0.5, objective=args.objective,
                    target_height_m=0.1,
                    shedding_fraction_per_m=args.shedding_fraction_per_m)
    model = HeightmapTransport()
    result = model.apply(terrain, stroke, blade)
    report = {
        'schema_version': 1, 'model': 'heightmap_path_transport',
        'calibrated': result.calibrated, 'input_kind': 'simulation_ground_truth',
        'scope': 'one straight stroke; not a completed job or machine execution',
        'grid': asdict(grid), 'blade': asdict(blade), 'stroke': asdict(stroke),
        'transport_parameters': {
            'repose_angle_deg': model.repose_angle_deg,
            'max_relax_sweeps': model.max_relax_sweeps,
        },
        'flags': result.flags, 'objective_complete': result.objective_complete,
        'initial_terrain_volume_m3': terrain.volume_m3,
        'final_terrain_volume_m3': result.terrain.volume_m3,
        'final_carried_volume_m3': result.carry.volume_m3,
        'cut_m3': sum(p.cut_m3 for p in result.steps),
        'released_m3': sum(p.released_m3 for p in result.steps),
        'shed_m3': sum(p.shed_m3 for p in result.steps),
        'spill_m3': result.spill_m3, 'remaining_cut_m3': result.remaining_cut_m3,
        'fill_deficit_m3': result.fill_deficit_m3, 'overfill_m3': result.overfill_m3,
        'mass_error_m3': result.mass_error_m3,
        'relaxation_converged': result.relaxation.converged,
        'relaxation_sweeps': result.relaxation.sweeps,
        'steps': [asdict(p) for p in result.steps],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'transport.npz', before_height_m=h,
                        after_height_m=result.terrain.height_m)
    (args.output/'transport.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('flags', 'objective_complete', 'mass_error_m3')}))
    return 0  # Experiment completed; goal completion is a separate report field.


if __name__ == '__main__':
    raise SystemExit(main())
