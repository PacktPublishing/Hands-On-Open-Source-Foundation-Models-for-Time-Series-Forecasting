"""Lag-Llama explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, CausalMask, Chip, Chips, Explainer, FanV, LagToken,
                       SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "

ex = Explainer(
    title="How Lag-Llama makes a forecast",
    subtitle="Rasul et al. · 2.5M parameters · LLaMA-style decoder on lagged values · zero-shot",
    steps=[
        Step("1. One series, with its dates", "yellow", [
            Text("Lag-Llama reads only the target's own past. A ? is a value not known yet.", space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False)], colors=["#2f2f2f"], space_after=0.12),
            Text("The dates matter: each step also carries six calendar features.", size=SIZE_H["small"],
                 color=GREY, space_after=0.06),
            Text("No covariates; each metal on its own.", size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("2. A token built from lags", "green", [
            Text("Values are scaled by the median and the interquartile range. The token for step t holds **84 "
                 "earlier values**, up to 1,093 steps back.", space_after=0.08),
            LagToken(),
            Text("The two statistics record the median and spread used for scaling, so the level is not lost.",
                 size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("3. Eight causal blocks", "orange", [
            Text("The last 32 tokens go through a small LLaMA-style decoder. RMSNorm in front, shortcut around.",
                 space_after=0.04),
            CausalMask(n=5, caption="earlier steps only"),
            Chips([Chip(["**Self-attention:** causal, 9 heads of 16 (width 144), RoPE"], align="left",
                        edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip([f"**Feed forward:** SwiGLU, 144{NB}→{NB}512{NB}→{NB}144"], align="left")], pad=0.05,
                  space_after=0.0),
        ]),
        Step("4. One step at a time", "blue", [
            Text("The last token gives a **Student-t distribution** for the next step. A value is sampled, appended, "
                 "and the model runs again.", space_after=0.08),
            FanV(h=0.62, samples=7, labels=("median", "sample paths")),
            Text("100 paths; their median is the point forecast. A 90-day horizon takes 90 passes.",
                 space_after=0.0),
        ]),
    ],
    footer=[("One of the first open forecasting foundation models: tiny, and built on lags rather than patches.",
             SIZE_H["foot"], "bold", INK),
            ("8 blocks · width 144 · 9 heads · 84 lags + 6 date features + 2 statistics per token · context of 32 "
             "tokens", SIZE_H["small"], "normal", GREY),
            ("In the notebooks: context of 32 tokens, linear RoPE scaling for longer horizons · each metal on its "
             "own · no covariates · 100 samples", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.15, 1.1, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "lag_llama_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
