"""Optional plotting adapter; core simulation has no plotting dependency."""
from pathlib import Path


def plot_demo(initial, final, report, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    b = initial.grid.bounds
    extent = [b.xmin_m, b.xmax_m, b.ymin_m, b.ymax_m]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    vmax = max(float(initial.height_m.max()), float(final.height_m.max()))
    for ax, terrain, title in zip(axes[:2], [initial, final], ["Initial synthetic pile", "After replanned material transfers"]):
        im = ax.imshow(terrain.height_m, extent=extent, origin="lower", vmin=0, vmax=vmax, cmap="cividis")
        for name, color in [("work", "white"), ("deposit", "tab:red")]:
            r = report["job"][name]
            ax.add_patch(Rectangle((r["xmin_m"], r["ymin_m"]), r["xmax_m"]-r["xmin_m"],
                                   r["ymax_m"]-r["ymin_m"], fill=False, edgecolor=color,
                                   linestyle="--", label=name))
        ax.set(title=title, xlabel="Site x (m)", ylabel="Site y (m)")
        ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=axes[:2], label="Height above flat ground (m)", shrink=0.8)
    values = [report["initial"]["work_excess_m3"]] + [h["work_excess_m3"] for h in report["history"]]
    axes[2].plot(range(len(values)), values, color="tab:blue", marker=".")
    axes[2].set(title="Excess in grading region", xlabel="Completed material transfers (count)",
                ylabel="Excess volume above target (m³)")
    axes[2].grid(alpha=0.2)
    fig.suptitle(f"D61EX-24 dimensions · simplified heightmap model · {report['passes']} transfers", fontsize=13)
    fig.text(0.01, -0.025, "Source: reproducible synthetic scenario; perfect terrain observations, prescribed blade, no vehicle or hydraulic dynamics.", fontsize=9)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_observation(truth, observation, output: Path):
    """Truth is provided separately for diagnostics, never read by an estimator."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = truth.grid.bounds
    extent = [b.xmin_m, b.xmax_m, b.ymin_m, b.ymax_m]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), layout="constrained")
    values = [truth.height_m, observation.height_m]
    finite = observation.height_m[observation.observed]
    vmin = min(0, float(finite.min())) if finite.size else 0
    vmax = max(float(truth.height_m.max()), float(finite.max()) if finite.size else 0, 0.1)
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("lightgray")
    for ax, height, title in zip(axes, values, ["Simulation truth (diagnostic only)", "Partial observation (gray = unknown)"]):
        im = ax.imshow(height, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        p = observation.sensor_pose
        ax.plot(p.x_m, p.y_m, marker="x", color="tab:red", markersize=8, label="Sensor XY")
        ax.set(title=title, xlabel="Site x (m)", ylabel="Site y (m)")
        ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=axes, label="Measured height above flat ground (m)", shrink=0.85)
    fig.suptitle(f"Idealized grid visibility · {int(observation.observed.sum())}/{truth.height_m.size} observed cells", fontsize=12)
    fig.text(0.01, -0.035, "Source: seeded synthetic scene; cell-top occlusion, vertical Gaussian noise and dropout. No sensor calibration or planner execution.", fontsize=8)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_estimate(observation, estimate, ground_elevation_m, output: Path):
    """Plot measured versus inferred excess heights without simulation truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    b = observation.grid.bounds
    extent = [b.xmin_m, b.xmax_m, b.ymin_m, b.ymax_m]
    measured = observation.height_m-ground_elevation_m
    predicted = estimate.predicted_height_m
    vmax = max(float(np.nanmax(measured)), float(predicted.max()), 0.1)
    cmap = plt.get_cmap("cividis").copy(); cmap.set_bad("lightgray")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), layout="constrained")
    for ax, h, title in zip(axes, [measured, predicted],
                            ["Measurements (gray = unknown)", f"{estimate.selected_model} model prediction (not measurements)"]):
        im = ax.imshow(h, extent=extent, origin="lower", vmin=0, vmax=vmax, cmap=cmap)
        ax.set(title=title, xlabel="Site x (m)", ylabel="Site y (m)")
    fig.colorbar(im, ax=axes, label="Height above supplied ground (m)", shrink=0.8)
    d = estimate.diagnostics
    fig.suptitle(f"{estimate.status} · total {d['total_volume_m3']:.2f} m³ · hidden {d['hidden_volume_m3']:.2f} m³", fontsize=12)
    fig.text(0.01, -0.035, "Source: saved partial scan only. Parametric inference; uncertainty diagnostics are conditional and not calibrated confidence bounds.", fontsize=8)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
