"""Kairos (50M) explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, Chip, Chips, EncDec, Explainer, FanV, SeriesRowsStacked,
                       SizeRouter, Step, Text, draw_explainer_h)

NB = " "

ex = Explainer(
    title="How Kairos makes a forecast",
    subtitle="ShanghaiTech University and Ant Group · 50M parameters · encoder-decoder with adaptive patch sizes · "
             "zero-shot",
    steps=[
        Step("1. One series at a time", "yellow", [
            Text("Kairos reads only the target's own past, up to 2,048 steps. A ? is a value not known yet.",
                 space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False)], colors=["#2f2f2f"], space_after=0.12),
            Text("No covariates; each metal on its own.", size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("2. Patch size per segment", "green", [
            Text("The scaled context is cut into 128-step segments. For each one a router keeps 3 of 5 options: "
                 "patches of 128, 64 or 32 steps, or a null that adds nothing.", space_after=0.08),
            SizeRouter(),
            Text("Busy stretches can get fine patches and smooth ones coarse patches; the kept sizes are blended "
                 "into tokens at the finest kept size.", size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("3. Encoder, then a decoder", "orange", [
            Text("Six encoder blocks read the patch tokens. Six decoder blocks start from two learned forecast "
                 "tokens and read the encoder through cross-attention.", space_after=0.04),
            EncDec(n_ctx=4, reg=True, top_label="patch tokens (encoder)", dec_labels=("future 1", "future 2"),
                   bottom_label="two learned forecast tokens"),
            Chips([Chip(["**Positions (DRoPE):** rotary, with frequencies tuned to each series' spectrum"],
                        align="left", edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Cross-attention:** forecast tokens read the encoder"], align="left",
                        edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
            Chips([Chip([f"**Feed forward:** 512{NB}→{NB}2,048{NB}→{NB}512, ReLU"], align="left")], pad=0.05,
                  space_after=0.0),
        ]),
        Step("4. 128 steps in one pass", "blue", [
            Text("A residual MLP turns each forecast token into **64 steps × 9 quantiles**, so one pass covers 128 "
                 "steps.", space_after=0.08),
            FanV(h=0.62),
            Text("Our runs also average with the forecast of −x and clip at zero a series that was never negative. "
                 "The median is the point forecast.", space_after=0.0),
        ]),
    ],
    footer=[("Kairos lets each series choose how finely it is cut into tokens, segment by segment.",
             SIZE_H["foot"], "bold", INK),
            ("6 + 6 blocks · width 512 · 8 heads · patch sizes 128, 64, 32 and 2 nulls, top 3 · 2 forecast tokens "
             "of 64 steps", SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 steps · each metal on its own · no covariates · every horizon in one pass",
             SIZE_H["small"], "normal", GREY)],
)

SHARES = [0.9, 1.2, 1.35, 1.05]
SHARES_2UP = [0.95, 1.15, 1.35, 1.05]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "kairos_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
