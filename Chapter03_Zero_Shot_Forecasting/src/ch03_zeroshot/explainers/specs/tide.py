"""TiDE baseline explainer (notebooks only)."""
import sys
from pathlib import Path

from ._baseline_common import NOTE_TUNE, inputs, trained
from ..explainer import ACCENT, GREY, INK, SIZE_H, Chip, Chips, DenseMap, Explainer, Step, Text, draw_explainer_h

ex = Explainer(
    title="How TiDE makes a forecast",
    subtitle="Das et al. (2023) · supervised baseline trained on each dataset · Darts TiDEModel",
    steps=[
        Step("1. A window of inputs", "yellow", inputs("400 days (36 months)")),
        Step("2. Covariates to 4 numbers", "green", [
            Text("A small residual network turns each step's covariates into 4 numbers, for the window and, for "
                 "known-ahead ones, the horizon too.", space_after=0.08),
            DenseMap(n_in=6, n_out=3, left_label="covariates", right_label="4 numbers", h=0.6),
            Text("This keeps the flattened input small when there are many covariates (seven past-only for temperature).", size=SIZE_H["small"],
                 color=GREY, space_after=0.0),
        ]),
        Step("3. Dense encoder and decoder", "orange", [
            Text("The window and the projected covariates are flattened into one long vector. Residual MLPs encode it "
                 "and decode 16 numbers per future step.", space_after=0.06),
            DenseMap(n_in=8, n_out=5, left_label="flattened input", right_label="per future step", h=0.7),
            Chips([Chip(["**Temporal decoder:** each step's 16 numbers plus that step's known-ahead covariates → the value"],
                        align="left", edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Linear shortcut:** the window mapped straight to the horizon and added"], align="left",
                        edge=ACCENT["orange"], lw=1.4)], pad=0.05, space_after=0.0),
        ]),
        Step("4. Train once, reuse", "blue", trained(NOTE_TUNE + " Hidden size 128 by default (tuned 32 to 128); 10 "
                                                     "epochs, MSE loss.")),
    ],
    footer=[("No attention and no recurrence: plain dense layers, which makes TiDE fast to train.",
             SIZE_H["foot"], "bold", INK),
            ("1 encoder and 1 decoder layer · window 400 days or 36 months · direct forecast of 90 days or 24 months · "
             "tuned: hidden size, learning rate", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.05, 1.25, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tide_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
