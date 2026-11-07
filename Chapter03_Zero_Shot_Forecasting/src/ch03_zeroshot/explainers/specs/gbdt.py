"""Gradient-boosted trees baseline explainer (LightGBM, XGBoost, CatBoost; notebooks only)."""
import sys
from pathlib import Path

from ._baseline_common import NOTE_TUNE, trained
from ..explainer import (ACCENT, GREY, INK, SIZE_H, Chip, Chips, Explainer, SeriesRowsStacked, Step, TableGrid, Text,
                       TreeRow, draw_explainer_h)

ex = Explainer(
    title="How the boosted trees make a forecast",
    subtitle="LightGBM, XGBoost and CatBoost on lagged values · supervised baselines trained on each dataset · Darts",
    steps=[
        Step("1. Lags become columns", "yellow", [
            Text("Each forecast origin becomes one row of a table: recent values of the target and covariates.",
                 space_after=0.08),
            TableGrid(cols=("y−1", "…", "y−14", "cov.", "y+h")),
            Text("Daily: the last 14 days (temperature also every 3rd day of the same season a year ago), past-only "
                 "covariates over 7 days, known-ahead ones every 10th future day. Monthly: 12 and 6 months.",
                 size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("2. Trees, one after another", "green", [
            Text("Boosting adds small decision trees one at a time; each new tree fits what the earlier ones still "
                 "get wrong.", space_after=0.08),
            TreeRow(caption="each tree fixes what is left"),
            Text("100 trees; tree size and learning rate tuned.", size=SIZE_H["small"], color=GREY,
                 space_after=0.0),
        ]),
        Step("3. Every future step at once", "orange", [
            Text("The forecast is direct: no step is fed back. The three libraries share out the horizon "
                 "differently.", space_after=0.06),
            Chips([Chip(["**LightGBM:** one model per future step and series"], align="left", edge=ACCENT["green"], lw=1.3)],
                  pad=0.05, space_after=0.05),
            Chips([Chip(["**XGBoost:** one booster; each round adds a tree for every step and series"], align="left",
                        edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**CatBoost:** one ensemble predicts every step and series together (MultiRMSE)"], align="left",
                        edge=ACCENT["orange"], lw=1.4)], pad=0.05, space_after=0.0),
        ]),
        Step("4. Train once, reuse", "blue", trained(NOTE_TUNE)),
    ],
    footer=[("Hand-picked lags and trees: the classic machine-learning answer to forecasting.",
             SIZE_H["foot"], "bold", INK),
            ("Prices as log differences · min-max scaled on the training rows · direct forecast of 90 days or 24 "
             "months · 100 trees", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.15, 1.0, 1.15, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "gbdt_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
