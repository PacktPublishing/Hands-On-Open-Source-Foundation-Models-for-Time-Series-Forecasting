"""TFT baseline explainer (notebooks only)."""
import sys
from pathlib import Path

from ._baseline_common import NOTE_TUNE, inputs, trained
from ..explainer import (ACCENT, GREY, INK, SIZE_H, Chip, Chips, Explainer, StateChain, Step, Text, VarWeights,
                       draw_explainer_h)

ex = Explainer(
    title="How the Temporal Fusion Transformer makes a forecast",
    subtitle="Lim et al. (2021) · supervised baseline trained on each dataset · Darts TFTModel",
    steps=[
        Step("1. A window of inputs", "yellow", inputs("120 days (36 months)",
                                                      note="Without known-ahead covariates a step counter is added "
                                                           "as one.")),
        Step("2. Pick the useful inputs", "green", [
            Text("At every step a small gated network gives each input a weight, so unhelpful inputs fade out.",
                 space_after=0.08),
            VarWeights(),
            Text("Past steps weigh all inputs; future steps weigh only the known-ahead ones.", size=SIZE_H["small"],
                 color=GREY, space_after=0.0),
        ]),
        Step("3. LSTM, then attention", "orange", [
            Text("An LSTM reads the past, a second one runs on over the horizon; then attention lets each future "
                 "step look back over the past and the future steps before it.", space_after=0.04),
            StateChain(n_past=4, n_future=2, top_label="selected inputs per step", state_label="LSTM encoder",
                       future_label="LSTM decoder"),
            Chips([Chip(["**Attention:** 2 heads that share one value projection, so the weights can be read"],
                        align="left", edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Gates and shortcuts** around every part; hidden size 16 (tuned 8 to 32)"],
                        align="left")], pad=0.05, space_after=0.0),
        ]),
        Step("4. Train once, reuse", "blue", trained(NOTE_TUNE + " Trained here with MSE for a point forecast, not "
                                                     "quantiles; 8 epochs.")),
    ],
    footer=[("Variable selection and attention make TFT one of the more readable deep forecasters.",
             SIZE_H["foot"], "bold", INK),
            ("1 LSTM layer · 2 attention heads · window 120 days or 36 months · direct forecast of 90 days or 24 "
             "months · tuned: hidden size, learning rate", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.05, 1.3, 0.9]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tft_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
