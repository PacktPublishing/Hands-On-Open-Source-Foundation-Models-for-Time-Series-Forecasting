"""TabPFN-TS explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, Chip, Chips, Explainer, FanV, SeriesRowsStacked, Step,
                       TableGrid, Text, draw_explainer_h)

ex = Explainer(
    title="How TabPFN-TS makes a forecast",
    subtitle="Prior Labs · TabPFN-v2 regressor · a tabular transformer used for forecasting · zero-shot",
    steps=[
        Step("1. One series, plus covariates", "yellow", [
            Text("TabPFN-TS reads one series and any covariates known ahead. A ? is a value not known yet.",
                 space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False), ("Known ahead", "day length", True)],
                              colors=["#2f2f2f", "#2e8a5f"], space_after=0.12),
            Text("Past-only covariates are not used; five metals become five separate tables.",
                 size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("2. Turn time into a table", "green", [
            Text("Each step becomes a row: a running index, calendar sine and cosine terms, detected seasonal terms "
                 "and the covariates. No lagged values.", space_after=0.08),
            TableGrid(cols=("index", "sin", "cos", "cov.", "y")),
            Text("Rows with y known are the training rows; the horizon rows have y = ?.", size=SIZE_H["small"],
                 color=GREY, space_after=0.0),
        ]),
        Step("3. Twelve two-way attention layers", "orange", [
            Text("The whole table goes in at once; the training set is just more input (in-context learning).",
                 space_after=0.06),
            Chips([Chip(["**Across features:** each cell reads the other cells in its row"], align="left",
                        edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Across rows:** each row reads the training rows; horizon rows never read each other"],
                        align="left", edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Feed forward** after each attention layer"], align="left")], pad=0.05, space_after=0.05),
            Text("Pre-trained only on synthetic tables, never on real time series.", size=SIZE_H["small"],
                 color=GREY, space_after=0.0),
        ]),
        Step("4. A distribution for each row", "blue", [
            Text("For each horizon row the output is a histogram over y; its quantiles and median are read off.",
                 space_after=0.08),
            FanV(h=0.62),
            Text("Several passes with shuffled preprocessing are averaged (2 on a CPU in our runs).",
                 size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
    ],
    footer=[("No forecasting architecture at all: a table model predicts the horizon rows from the past rows.",
             SIZE_H["foot"], "bold", INK),
            ("12 layers · attention across features and across rows · output: a discretised distribution per row",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: ungated TabPFN-v2 weights · last 2,048 days or full monthly history · calendar "
             "covariates for D2 · median as the point", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.05, 1.1, 1.2, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tabpfn_ts_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
