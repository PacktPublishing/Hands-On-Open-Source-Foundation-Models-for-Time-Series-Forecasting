"""DLinear baseline explainer (notebooks only)."""
import sys
from pathlib import Path

from ._baseline_common import NOTE_TUNE, inputs, trained
from ..explainer import ACCENT, GREY, INK, SIZE_H, Chip, Chips, Decomp, DenseMap, Explainer, Step, Text, draw_explainer_h

ex = Explainer(
    title="How DLinear makes a forecast",
    subtitle="Zeng et al. (2023) · supervised baseline trained on each dataset · Darts DLinearModel",
    steps=[
        Step("1. A window of inputs", "yellow", inputs("400 days (36 months)")),
        Step("2. Split trend and remainder", "green", [
            Text("A moving average over the window gives each row its trend; what is left is the remainder.",
                 space_after=0.08),
            Decomp(),
            Text("Average length: tuned among 7, 25 and 61 days (3, 7 and 13 months).", size=SIZE_H["small"],
                 color=GREY, space_after=0.0),
        ]),
        Step("3. Two linear layers", "orange", [
            Text("One linear layer maps the whole trend window to the horizon, another the remainder; the two "
                 "outputs are added.", space_after=0.06),
            DenseMap(left_label="window", right_label="horizon"),
            Chips([Chip([r"$\hat{y}=W_T\,T+W_R\,R+W_x\,x$"], math=True)], pad=0.04, space_after=0.06),
            Text("T and R are the trend and remainder of every input row; x is the known-ahead covariates at each future step.",
                 size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("4. Train once, reuse", "blue", trained(NOTE_TUNE + " 15 epochs, MSE loss.")),
    ],
    footer=[("A strong, simple yardstick: if a foundation model cannot beat two linear layers, it adds little.",
             SIZE_H["foot"], "bold", INK),
            ("Window 400 days or 36 months · direct forecast of 90 days or 24 months at once · tuned: average length, "
             "learning rate", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.05, 1.15, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "dlinear_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
