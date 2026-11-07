"""N-HiTS baseline explainer (notebooks only)."""
import sys
from pathlib import Path

from ._baseline_common import NOTE_TUNE, inputs, trained
from ..explainer import (ACCENT, GREY, INK, SIZE_H, Chip, Chips, CoarseInterp, Explainer, PooledRows, Step, Text,
                       draw_explainer_h)

NB = " "

ex = Explainer(
    title="How N-HiTS makes a forecast",
    subtitle="Challu et al. (2023) · supervised baseline trained on each dataset · Darts NHiTSModel",
    steps=[
        Step("1. A window of inputs", "yellow", inputs("400 days (36 months)", roles=("Target", "Past only"),
                                                      note="N-HiTS takes past-only covariates; day length is not "
                                                           "used.")),
        Step("2. Three stacks, three time scales", "green", [
            Text("Each stack first max-pools the window: the first sees only the broad shape, the last every step.",
                 space_after=0.08),
            PooledRows(),
            Text("Daily pooling sizes 200, 14 and 1 steps (monthly 18, 4, 1).", size=SIZE_H["small"], color=GREY,
                 space_after=0.0),
        ]),
        Step("3. A few points, then interpolate", "orange", [
            Text("In each stack an MLP predicts only a few forecast points, and straight lines join them up to "
                 "every step.", space_after=0.06),
            CoarseInterp(),
            Chips([Chip([f"**MLP:** 2 layers, width 512 (tuned 128 to 512), ReLU"], align="left",
                        edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Points per stack:** 2, 15 and 90 for 90 days; each stack also returns the part of the "
                         "window it explained, which is removed before the next"], align="left")], pad=0.05,
                  space_after=0.0),
        ]),
        Step("4. Add the stacks, train once", "blue",
             trained("The forecast is the sum of the three stacks. " + NOTE_TUNE + " 10 epochs, MSE loss.")),
    ],
    footer=[("Coarse stacks capture the slow movements cheaply; the last stack fills in the detail.",
             SIZE_H["foot"], "bold", INK),
            ("3 stacks of 1 block · 2 layers each · window 400 days or 36 months · direct forecast of 90 days or 24 "
             "months · tuned: width, learning rate", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.05, 1.25, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "nhits_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
