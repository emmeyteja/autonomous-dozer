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
