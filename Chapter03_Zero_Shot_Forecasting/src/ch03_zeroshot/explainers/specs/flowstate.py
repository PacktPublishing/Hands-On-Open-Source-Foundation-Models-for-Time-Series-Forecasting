"""FlowState r1.1 explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, BasisCurves, Chip, Chips, Explainer, FanV, SeriesRowsStacked,
                       StateChain, Step, Text, draw_explainer_h)

NB = " "

ex = Explainer(
    title="How FlowState makes a forecast",
    subtitle="IBM · 18.5M parameters (r1.1) · state-space encoder, functional decoder · zero-shot",
    steps=[
        Step("1. Every step a token", "yellow", [
            Text("FlowState reads one series. There are no patches: each step becomes its own token.",
                 space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False)], colors=["#2f2f2f"], space_after=0.12),
            Text("Each value is scaled with the mean and standard deviation of the steps up to it, then a linear "
                 "layer turns value and missing flag into 512 numbers.", size=SIZE_H["small"], color=GREY,
                 space_after=0.0),
        ]),
        Step("2. A clock for any rate", "green", [
            Text("The scale factor s tells the model how many steps make one season, so the same weights fit "
                 "daily or monthly data.", space_after=0.06),
            Chips([Chip([r"$s=\dfrac{24}{m}$"], math=True)], pad=0.04, space_after=0.08),
            Text("m is the number of steps in one season: 365 for days, 12 for months, 252 for trading days.",
                 size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("3. Six state-space layers", "orange", [
            Text("No attention. Each S5 layer updates a hidden state at every step; its step size Δ is multiplied "
                 "by s.", space_after=0.06),
            Chips([Chip([r"$h_t=\bar{A}\,h_{t-1}+\bar{B}\,x_t$"], math=True)], pad=0.04, space_after=0.06),
            StateChain(n_past=6, n_future=0, top_label="one token per step", state_label="state updated every step"),
            Chips([Chip(["**S5 layer:** state of 512, then a sigmoid output gate"], align="left",
                        edge=ACCENT["green"], lw=1.4)], pad=0.05, space_after=0.05),
            Chips([Chip([f"**Feed forward:** gated, 512{NB}→{NB}1,024{NB}→{NB}512, then LayerNorm; shortcut "
                         "around the layer"], align="left")], pad=0.05, space_after=0.0),
        ]),
        Step("4. A curve, not fixed steps", "blue", [
            Text("The decoder gives the weights of **256 Legendre curves** for each of **9 quantiles**. The sum is "
                 "read off at each future step.", space_after=0.08),
            BasisCurves(h=0.5),
            FanV(h=0.5),
            Text("The point forecast is a weighted mean of the 9 quantiles; the middle ones count most.",
                 space_after=0.0),
        ]),
    ],
    footer=[("A continuous-time encoder and decoder let one set of weights serve any sampling rate.",
             SIZE_H["foot"], "bold", INK),
            ("6 S5 layers · width 512 · state 512 · 256 Legendre curves per quantile · trained on 4,096-step contexts",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · s from the yearly season · each metal on its "
             "own · no covariates", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 0.95, 1.3, 1.1]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "flowstate_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
