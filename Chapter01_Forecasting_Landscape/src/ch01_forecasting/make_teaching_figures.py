"""Reproduce Chapter 1's original synthetic illustrations.

No panel contains measured ETTh1 data or Forecasting Zoo model results.
Run with ``python scripts/make_teaching_figures.py`` from the chapter folder.
New vector PDFs, 220-dpi PNGs and a manifest go into a fresh run directory;
the bundled reference figures are never replaced.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from scipy.stats import norm

from .artifacts import fresh_output_dir, utc_now

ROOT = Path(__file__).resolve().parents[2]
SEED = 41705
BLUE = "#176B91"
ORANGE = "#A65712"
GREEN = "#19765A"
GREY = "#64696F"
INK = "#202B35"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.6,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GREY,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "axes.grid": True,
    "grid.color": "#E4E7E9",
    "grid.linewidth": .45,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def save(fig, name, details, *, output_dir, manifest):
    """Write one generated figure, after the caller has reserved fresh output."""
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"{name}.{extension}", dpi=220,
                    bbox_inches="tight", pad_inches=.05)
    plt.close(fig)
    manifest["figures"][name] = details


def decomposition(*, output_dir, manifest):
    rng = np.random.default_rng(SEED)
    t = np.arange(240)
    trend = 18 + .035 * t
    seasonal = 2.5 * np.sin(2 * np.pi * t / 24)
    remainder = rng.normal(0, .65, len(t))
    observed = trend + seasonal + remainder
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 3.4), sharex=True, layout="constrained")
    content = [(observed, "(a) Observed: all components", INK),
               (trend, "(b) Known generating trend", BLUE),
               (seasonal, "(c) Known 24-hour seasonal component", ORANGE),
               (remainder, "(d) Known random remainder", GREEN)]
    for ax, (values, title, color) in zip(axes.flat, content):
        ax.plot(t, values, color=color, linewidth=.85)
        ax.set_title(title, loc="left", pad=5)
        ax.set_ylabel("Synthetic units")
        ax.set_xlim(0, 239)
        ax.set_xticks([0, 48, 96, 144, 192, 239])
    for ax in axes[1]:
        ax.set_xlabel("Hour")
    save(fig, "decomposition", {
        "caption": "A synthetic additive series and its known generating components: y_t = m_t + s_t + r_t. These are specified components, not an estimated decomposition of ETTh1.",
        "alt_text": "Four panels show a rising noisy series, its linear trend, a repeating daily sine wave, and zero-centered random noise.",
        "parameters": {"n": 240, "trend": "18 + 0.035*t", "seasonality": "2.5*sin(2*pi*t/24)", "remainder": "iid Normal(0,0.65^2)"},
        "dimensions_inches": [5.8, 3.4],
    }, output_dir=output_dir, manifest=manifest)


def stationarity(*, output_dir, manifest):
    rng = np.random.default_rng(SEED + 1)
    n = 240
    eps = rng.normal(size=n)
    phi = .7
    stable = np.empty(n)
    stable[0] = rng.normal(0, 1 / np.sqrt(1 - phi**2))
    for t in range(1, n):
        stable[t] = phi * stable[t - 1] + eps[t]
    walk = np.cumsum(rng.normal(size=n))
    sigma = np.where(np.arange(n) < 120, .7, 2.4)
    changing_variance = sigma * rng.normal(size=n)
    fig, axes = plt.subplots(3, 1, figsize=(5.8, 3.75), sharex=True, layout="constrained")
    for ax, vals, title, color in zip(axes, [stable, walk, changing_variance],
        [r"(a) Stationary AR(1): $y_t=0.7y_{t-1}+\epsilon_t$",
         r"(b) Random walk: $y_t=y_{t-1}+\epsilon_t$",
         r"(c) Constant mean, changing variance: $y_t=\sigma_t\epsilon_t$"], [BLUE, ORANGE, GREEN]):
        ax.plot(np.arange(n), vals, color=color, lw=.8)
        ax.axhline(0, color=GREY, lw=.6, ls=":")
        ax.set_title(title, loc="left", pad=4)
        ax.set_ylabel("Value")
        ax.set_xlim(0, n - 1)
    axes[2].axvline(119.5, color=INK, ls="--", lw=.7)
    axes[2].text(123, axes[2].get_ylim()[1] * .66, "variance increases", fontsize=7.5)
    axes[2].set_xlabel("Time step")
    save(fig, "stationarity", {
        "caption": "Three synthetic processes separate stable dependence from nonstationarity. Panel (a) is initialized from its stationary distribution; the random walk has variance that grows with time; panel (c) changes variance halfway through. Appearance alone cannot establish stationarity.",
        "alt_text": "Three time plots compare a stable autoregression oscillating around zero, a wandering random walk, and zero-mean noise that becomes much more variable halfway through.",
        "parameters": {"n": n, "phi": phi, "ar_innovation_sd": 1, "ar_initial_distribution": "Normal(0,1/(1-phi^2))", "random_walk_innovation_sd": 1, "variance_change_index": 120, "sigma_before": .7, "sigma_after": 2.4},
        "dimensions_inches": [5.8, 3.75],
    }, output_dir=output_dir, manifest=manifest)


def uncertainty(*, output_dir, manifest):
    rng = np.random.default_rng(SEED + 2)
    # A drifted Gaussian random walk permits exact marginal interval calculation.
    origin, horizon = 96, 48
    drift, sd = .06, .4
    history = 20 + np.cumsum(drift + rng.normal(0, sd, origin))
    h = np.arange(1, horizon + 1)
    mean = history[-1] + drift * h
    scale = sd * np.sqrt(h)
    future = history[-1] + np.cumsum(drift + rng.normal(0, sd, horizon))
    fig, ax = plt.subplots(figsize=(5.8, 2.6), layout="constrained")
    t_future = origin - 1 + h
    ax.fill_between(t_future, mean - norm.ppf(.975) * scale,
                    mean + norm.ppf(.975) * scale, color=BLUE, alpha=.12,
                    linewidth=0, label="95% pointwise interval")
    ax.fill_between(t_future, mean - norm.ppf(.90) * scale,
                    mean + norm.ppf(.90) * scale, color=BLUE, alpha=.26,
                    linewidth=0, label="80% pointwise interval")
    ax.plot(np.arange(origin), history, color=INK, lw=.9, label="Observed history")
    ax.plot(t_future, mean, color=BLUE, lw=1.25, ls="--", label="Forecast median")
    ax.plot(t_future, future, color=ORANGE, lw=.9, label="One future realization")
    ax.axvline(origin - 1, color=GREY, ls=":", lw=.9)
    ax.text(origin - 2, ax.get_ylim()[0] + .5, r"Origin $T$", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel(r"Time step; future $T+1,\ldots,T+H$")
    ax.set_ylabel("Synthetic units")
    ax.set_xlim(48, origin - 1 + horizon)
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.9, columnspacing=1)
    save(fig, "uncertainty", {
        "caption": "Exact marginal Gaussian prediction intervals for a synthetic random walk with known drift and innovation variance. One realized path cannot establish calibration, and pointwise 95% intervals do not imply 95% simultaneous path coverage.",
        "alt_text": "A historical series ends at an origin, after which a dashed median rises with two widening shaded bands and one realized future path.",
        "parameters": {"history_length": origin, "display_history_start": 48, "horizon": horizon, "drift": drift, "innovation_sd": sd, "forecast_mean": "y_T + 0.06*h", "forecast_sd": "0.4*sqrt(h)", "interval_levels": [80, 95]},
        "dimensions_inches": [5.8, 2.6],
    }, output_dir=output_dir, manifest=manifest)


def box(ax, xy, width, height, text, face, border=GREY, fontsize=8.4):
    ax.add_patch(FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.015,rounding_size=0.02", linewidth=.8,
                              edgecolor=border, facecolor=face))
    ax.text(xy[0] + width/2, xy[1] + height/2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                linewidth=.85, color=GREY))


def learning_regimes(*, output_dir, manifest):
    fig, ax = plt.subplots(figsize=(5.8, 3.0), layout="constrained")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    row_y = [.76, .43, .10]
    labels = ["Local", "Global", "Pretrained"]
    descriptions = [
        ("One series", "Fit one model\nper series", "Forecast\nthat series"),
        ("Training panel\nof many series", "Fit shared\nparameters", "Forecast panel\nusing context"),
        ("Broad pretraining\ncorpus", "Reusable\ncheckpoint", "New context\n+ frozen weights"),
    ]
    for y, label, texts in zip(row_y, labels, descriptions):
        ax.text(0, y + .092, label, ha="left", va="center", fontsize=8.6, weight="bold")
        for x, text, face in zip([.18, .47, .76], texts, ["#F1F3F4", "#E3EEF2", "#EDF2EF"]):
            box(ax, (x, y), .225, .18, text, face)
        arrow(ax, (.408, y + .09), (.463, y + .09))
        arrow(ax, (.698, y + .09), (.753, y + .09))
    ax.text(.585, .021, "Optional fine-tuning updates the checkpoint using target-domain data.",
            ha="center", va="bottom", fontsize=7.4, color=GREY)
    save(fig, "learning_regimes", {
        "caption": "Where statistical strength is shared: a local fit uses one series, a global fit learns across a panel, and a pretrained checkpoint can be reused with new context. The last row shows zero-shot inference; fine-tuning adds task-specific weight updates. These are learning regimes, not exclusive architecture classes.",
        "alt_text": "Three rows compare one-series local fitting, shared training across many series, and broad-corpus pretraining followed by use of a reusable checkpoint on new context.",
        "parameters": {"type": "conceptual diagram", "empirical_values": False},
        "dimensions_inches": [5.8, 3.0],
    }, output_dir=output_dir, manifest=manifest)


def main(argv=None):
    """Generate the four illustrations in a fresh directory and return its path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="Fresh directory for figures (default: a new runs/teaching_*/figures).")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        output_dir = fresh_output_dir(root=ROOT, prefix="teaching") / "figures"
        output_dir.mkdir()
    else:
        output_dir = fresh_output_dir(args.output_dir, root=ROOT, prefix="teaching")
    manifest = {
        "seed": SEED,
        "status": "executed successfully",
        "execution_utc": utc_now(),
        "provenance": "Original code-generated synthetic illustrations; no sample-book artwork or empirical model results.",
        "command": "python scripts/make_teaching_figures.py",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "matplotlib": matplotlib.__version__},
        "figures": {},
    }
    for draw in (decomposition, stationarity, uncertainty, learning_regimes):
        draw(output_dir=output_dir, manifest=manifest)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Generated four original synthetic/conceptual figures in", output_dir)
    return output_dir


if __name__ == "__main__":
    main()
