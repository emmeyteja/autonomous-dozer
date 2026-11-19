# First simulation milestone

## Implemented module boundaries

- `domain`: immutable terrain snapshots, grid geometry, blade specification, job and version-bound material-pass contracts. SI units; rows represent y, columns x; grid origin is the lower-left cell edge; heights are above known flat ground.
- `terrain`: conical, stadium-ridge and reproducible multi-dump pile generators.
- `planning`: deterministic next-pass selection from perfect simulated terrain. It selects the largest-volume lane at a top-down cut height, limits lane width to blade width and rejects unsupported geometry/push distance.
- `simulation`: capacity-limited +x removal, uniform placement into a separate deposition area, and mass-conserving cardinal-neighbor repose relaxation with closed map boundaries.
- `profiles`: manufacturer dimensions and separate scenario assumptions.
- `cli` and `visualization`: scenario orchestration, JSON/NPZ artifacts and optional plots. Domain algorithms do not depend on ROS, plotting or manufacturer APIs.

Each transfer creates a new terrain version; commands targeting another version are rejected. Replanning occurs after each transfer and relaxation. A sweep limit is reported as non-convergence rather than silently accepted. The demo bounds grid size and pass count, never overwrites an existing output directory, and exits nonzero if it reaches a limit or fails to make progress.

## What the demonstration means

The supported task is removing material above a target elevation from a rectangular work region into a disjoint, explicitly chosen deposit region ahead in +x. It does not yet spread material to fill below-grade cells. `upper_excess_within_tolerance` means only the upper-excess condition was met. RMSE and fraction within grade tolerance are reported separately so positive target elevations do not turn unfilled ground into a false grading success.

Each pass is an abstract material transfer. Material is removed in column order until capacity is reached, then uniformly placed in the designated deposit region. A partial last column approximates stopping within a cell. The simulator does not model how a moving blade carries and sheds material along that path. Slumping checks cardinal neighbors only; it is an anisotropic numerical approximation, not a calibrated granular-material solver.

The cut height is prescribed for each simulated pass. This permits experimenting with layers but does not establish that the installed Trimble system accepts intermediate layer targets. The planner verifies material regions, not a vehicle's full swept footprint. It does not produce collision-checked trajectories or actuator commands.

## Next development gates

1. Add visibility/noise and a pile estimator, keeping unknown terrain separate from simulation truth. Validate volume and hidden-side errors on held-out shapes.
2. Replace broad deposit placement with calibrated blade carrying/spreading rules. Add feasible disposal volume, slopes, blade approach and full vehicle swept-footprint checks.
3. Add tracked kinematics, reverse tracking, speed/stop constraints and vehicle-specific adapters. Profile metadata is not yet a dynamics model.
4. Add supervisor and command gate before learned-policy integration. Policy failure may select a valid deterministic candidate; shared sensing/localization/actuator failure must cause pause/stop and recovery.
5. Run learned policies in shadow mode against the same work-segment contract. Validate transitions, not only policy reward.
6. Retain external Trimble blade authority in real-machine trials. Future cylinder feedback belongs behind a separate blade adapter.

## Autoware integration boundary

No custom ROS messages are introduced in this milestone. Core algorithms run without ROS. Later use `autoware_planning_msgs/Trajectory` for vehicle trajectories where semantics fit and `nav_msgs/Odometry` for state. Keep earthwork phases and blade intent in a distinct work-level contract. Upstream [Autoware Lateral](https://github.com/autowarefoundation/autoware_msgs/blob/main/autoware_control_msgs/msg/Lateral.msg) represents Ackermann tire steering; do not encode tracked yaw rate as tire angle. A tracked vehicle requires a documented adapter or appropriate velocity/yaw-rate contract.

No proprietary autonomy-platform source is included. Modules in this repository are independently implemented, with public manufacturer references recorded separately.
