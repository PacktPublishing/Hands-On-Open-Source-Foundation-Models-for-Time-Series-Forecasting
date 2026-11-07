"""Shared figure style and the wavelet-coherence map (all notebooks)."""
# --- cell: plotting ---
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#f08a89", "#b5302f"])


def set_style():
    """Quiet axes, light grid, thin lines: the chart style used in every notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.4, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": True,
        "legend.facecolor": SURFACE, "legend.edgecolor": "none", "legend.framealpha": 0.85,
        "axes.prop_cycle": mpl.cycler(color=SERIES), "figure.dpi": 110,
        # DejaVu Sans ships with matplotlib: the same font on Colab, a Mac or Linux
        "font.family": ["DejaVu Sans"], "font.sans-serif": ["DejaVu Sans"],
    })


def plot_coherence(ax, res, times, title="", period_unit="months", arrow_every=None,
                   period_ticks=None, show_colorbar=False):
    """Wavelet-coherence map in the layout of biwavelet::plot.biwavelet.

    Colour: coherence R^2 (0 to 1). Black outline: FDR-significant regions.
    Faded area: outside the cone of influence. Arrows (drawn inside the
    significant regions): right = in phase, left = anti-phase, up = the
    exogenous (first) series leads, down = the target leads.
    """
    import matplotlib.dates as mdates
    import pandas as pd

    t = mdates.date2num(pd.DatetimeIndex(times).to_pydatetime())
    logp = np.log2(res["period"])
    mesh = ax.contourf(t, logp, res["rsq"], levels=np.linspace(0, 1, 11), cmap=SEQUENTIAL,
                       vmin=0, vmax=1, extend="neither")
    sig = (res["sig"] & res["inside"]).astype(float)
    if sig.any():
        ax.contour(t, logp, sig, levels=[0.5], colors=INK, linewidths=1.1)
    coi = np.log2(np.maximum(res["coi"], res["period"].min()))
    ax.fill_between(t, coi, logp.max(), color=SURFACE, alpha=0.55, lw=0)
    ax.plot(t, coi, color=INK_2, lw=0.8, ls="--")
    # phase arrows inside the significant regions only, about 0.25 inch apart, centred in each region
    n_t, n_s = len(t), len(logp)
    box = ax.get_position()
    width_in, height_in = box.width * ax.figure.get_figwidth(), box.height * ax.figure.get_figheight()
    step_t = arrow_every or max(1, int(n_t * 0.25 / width_in))
    gap = max(1, int(np.ceil(n_s * 0.25 / height_in)))
    mask = res["sig"] & res["inside"]
    rows, cols = [], []
    for j in range(step_t // 2, n_t, step_t):
        runs = np.split(np.flatnonzero(mask[:, j]), np.flatnonzero(np.diff(np.flatnonzero(mask[:, j])) > 1) + 1)
        for run in runs:
            if run.size:
                offset = (run.size - 1) % gap // 2
                rows += list(run[offset::gap])
                cols += [j] * len(run[offset::gap])
    if rows:
        ph = res["phase"][rows, cols]
        ax.quiver(t[cols], logp[rows], np.cos(ph), np.sin(ph), color=INK, angles="uv",
                  scale_units="inches", scale=1 / 0.16, units="inches", width=0.012,
                  headwidth=4, headlength=4.5, headaxislength=4, pivot="middle")
    ax.set_ylim(logp.max(), logp.min())  # short periods at the top, as in biwavelet
    ticks = period_ticks or [2 ** k for k in range(int(np.ceil(logp.min())), int(logp.max()) + 1)]
    ax.set_yticks(np.log2(ticks))
    ax.set_yticklabels([f"{p:g}" for p in ticks])
    ax.set_ylabel(f"Period ({period_unit})")
    ax.xaxis_date()
    ax.set_title(title, loc="left")
    ax.grid(False)
    if show_colorbar:
        plt.colorbar(mesh, ax=ax, label="Coherence $R^2$", ticks=[0, 0.5, 1])
    return mesh
