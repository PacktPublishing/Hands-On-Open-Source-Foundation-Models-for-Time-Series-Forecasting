"""TimesFM-3 explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, AttentionGridC, Chip, Chips, Explainer, FanV, PatchLookahead,
                       SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "

ex = Explainer(
    title="How TimesFM-3 makes a forecast",
    subtitle="Google · 331M parameters · decoder-only transformer · zero-shot",
    steps=[
        Step("1. One row per series", "yellow", [
            Text("Up to 32 rows: targets, past-only and known-ahead covariates. A ? is a value not known yet.",
                 space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False), ("Known ahead", "day length", True),
                               ("Past only", "NAO index", False)],
                              colors=["#2f2f2f", "#2e8a5f", "#9a7b33"], space_after=0.12),
            Text("Five metal prices go in together as five target rows.", size=SIZE_H["small"], color=GREY,
                 space_after=0.0),
        ]),
        Step("2. Scale, then build tokens", "green", [
            Text("Each row is scaled as it goes, with the mean and standard deviation up to each patch. A token "
                 "holds a patch of 32 steps and the next 64 steps, which only known-ahead rows can fill.", space_after=0.08),
            PatchLookahead(token_label="one token: 1,280 numbers", space_after=0.1),
            Text("A row with a strong straight-line trend is detrended first.", size=SIZE_H["small"], color=GREY,
                 space_after=0.0),
        ]),
        Step("3. Twenty blocks, two kinds of attention", "orange", [
            Text("RMSNorm sits before and after every layer, with a shortcut around it. Future patches go in as "
                 "masked tokens.", space_after=0.04),
            AttentionGridC(["temperature", "day length", "NAO index"], reg=False, causal=True, fill_rows=[1]),
            Chips([Chip(["**Sequence attention:** along the row, this patch and earlier ones (RoPE)"],
                        align="left", edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Variate attention:** down the column, every row at the same patch"],
                        align="left", edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Feed forward:** each token on its own, width 1,280, ReLU"], align="left")], pad=0.05,
                  space_after=0.0),
        ]),
        Step("4. Every horizon in one pass", "blue", [
            Text("A linear head turns each future token of a target row into **64 steps × 9 quantiles**.",
                 space_after=0.08),
            FanV(h=0.62),
            Text("Tokens 32 steps apart overlap by 32 steps; the overlaps are blended. The trend and scaling are "
                 "undone and the median is the point forecast.", space_after=0.0),
        ]),
    ],
    footer=[("Variate attention lets each target read its covariates and the other targets at every patch.",
             SIZE_H["foot"], "bold", INK),
            ("20 blocks · width 1,280 · patches of 32 in, 64 out · up to 32 rows · context up to 15,360 steps",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · all covariates and the five metals "
             "together · weights for non-commercial use", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.05, 1.1, 1.35, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "timesfm3_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
