"""Figure of Chapter 3: what the covariates change, drawn from the stored results of Notebook 11.

Usage (from the chapter folder, in the dataprep environment):
    .venv-dataprep/bin/python scripts/make_covariate_figure.py [output folder]

It writes fig_ch03_covariate_gains.pdf and .png to results/figures/ unless you give another folder.
Nothing is recomputed: the gains come from results/tables/scores_per_forecast.csv through the same
functions Notebook 11 uses (ch03_zeroshot.report.covariate_gain, with ARIMAX, ARDL and VARX compared
with their plain twins ARIMA, AR and VAR).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]          # the chapter folder
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
from ch03_zeroshot.capabilities import ZERO_SHOT_MODELS   # noqa: E402
from ch03_zeroshot.metrics import aggregate               # noqa: E402
from ch03_zeroshot.report import TWINS, covariate_gain    # noqa: E402
from ch03_zeroshot.tables import display_names            # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 6.4, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
INK, MUTED = "#0b0b0b", "#52514e"
SLOT = ["#2a78d6", "#eb6834"]          # foundation models blue, baselines orange
NAMES = display_names()
FMS = {r[0] for r in ZERO_SHOT_MODELS}

agg = aggregate(pd.read_csv(REPO / "results" / "tables" / "scores_per_forecast.csv"))
d = agg.copy()
for rich, plain in TWINS.items():      # a model that needs covariates is compared with its plain twin
    twin = agg[(agg["model"] == plain) & agg["dataset"].isin(["D1", "D3"])].assign(model=rich)
    d = pd.concat([d[~((d["model"] == rich) & d["dataset"].isin(["D1", "D3"]))], twin], ignore_index=True)
gain = covariate_gain(d, "MASE")
gain = gain[gain["horizon"].notna()]

fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.5))
panels = [("D2 vs D1", "Temperature: twelve covariates (D2 against D1)"),
          ("D4 vs D3", "Metals: three past-only drivers (D4 against D3)")]
for ax, (comp, title) in zip(axes, panels):
    g = gain[gain["comparison"] == comp].groupby("model")["change_%"].agg(["mean", "min", "max"])
    fm = g[g.index.isin(FMS)].sort_values("mean")
    bl = g[~g.index.isin(FMS)].sort_values("mean")
    rows = list(fm.index) + [None] + list(bl.index)
    y = {m: len(rows) - 1 - i for i, m in enumerate(rows) if m is not None}
    for m, yi in y.items():
        colour = SLOT[0] if m in FMS else SLOT[1]
        ax.plot([g.loc[m, "min"], g.loc[m, "max"]], [yi, yi], color=colour, lw=1.4, alpha=0.55,
                solid_capstyle="round")
        ax.plot(g.loc[m, "mean"], yi, "o", color=colour, ms=3.8, markeredgecolor="white", markeredgewidth=0.6)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(list(y.values()), [NAMES.get(m, m).replace(" (vs ", " vs ") for m in y])
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=7.2, color=INK)
    ax.set_xlabel("% change in MASE with the covariates")
    print(comp, g["mean"].round(1).to_dict())
from matplotlib.lines import Line2D                    # noqa: E402
fig.legend(handles=[Line2D([], [], color=SLOT[0], marker="o", ms=3.8, lw=1.4, label="Foundation model"),
                    Line2D([], [], color=SLOT[1], marker="o", ms=3.8, lw=1.4, label="Baseline (tuned)")],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.text(0.5, -0.02, "Dot: mean over the five horizons; line: range over the horizons. Left of zero, the covariates helped.",
         ha="center", fontsize=6.4, color=MUTED)
fig.tight_layout()
fig.savefig(OUT / "fig_ch03_covariate_gains.pdf", metadata={"CreationDate": None})   # same bytes on every run
fig.savefig(OUT / "fig_ch03_covariate_gains.png", dpi=200)
print("wrote", OUT / "fig_ch03_covariate_gains.pdf")
