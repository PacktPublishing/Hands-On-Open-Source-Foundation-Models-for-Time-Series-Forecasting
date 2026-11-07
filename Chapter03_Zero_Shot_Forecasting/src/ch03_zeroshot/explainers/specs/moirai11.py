"""Moirai-1.1 (small) explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, BlockAttention, Chip, Chips, Explainer, FanV, FlatSequence,
                       MixtureCurve, PatchToTokenV, SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "

ex = Explainer(
    title="How Moirai-1.1 makes a forecast",
    subtitle="Salesforce · 14M parameters (small) · masked encoder with any-variate attention · zero-shot",
    steps=[
        Step("1. All rows in one sequence", "yellow", [
            Text("Targets and covariates are laid end to end as one long sequence. A ? is a value not known yet.",
                 space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False), ("Known ahead", "day length", True),
                               ("Past only", "NAO index", False)],
                              colors=["#2f2f2f", "#2e8a5f", "#9a7b33"], space_after=0.1),
            FlatSequence(),
            Text("Any number of rows fits, so the five metals share one sequence.", size=SIZE_H["small"],
                 color=GREY, space_after=0.0),
        ]),
        Step("2. Scale, patch, mask the future", "green", [
            Text("Each row is scaled by the mean μ and standard deviation σ of its own past.", space_after=0.06),
            Chips([Chip([r"$\tilde{x}=\dfrac{x-\mu}{\sigma}$"], math=True)], pad=0.04, space_after=0.1),
            PatchToTokenV(patch_len=32, rows=("value",), caption="one patch = 32 steps",
                          token_label="one token: 384 numbers"),
            Text("Patch sizes 8 to 128 each have their own input layer; our runs use 32. Future target patches "
                 "become a learned mask token M.", size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("3. Six blocks of any-variate attention", "orange", [
            Text("Every token reads every other token, past and future, in every row. RMSNorm in front, shortcut "
                 "around.", space_after=0.06),
            BlockAttention(),
            Chips([Chip(["**Time:** rotary positions (RoPE) along each row"], align="left", edge=ACCENT["blue"],
                        lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Rows:** one learned bias for same row, one for other row; no row order"],
                        align="left", edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
            Chips([Chip([f"**Feed forward:** SwiGLU, 384{NB}→{NB}1,024{NB}→{NB}384"], align="left")], pad=0.05,
                  space_after=0.0),
        ]),
        Step("4. A mixture, then samples", "blue", [
            Text("Each mask token gives, for each of its 32 steps, a **mix of four distributions**: Student-t, "
                 "normal, negative binomial, log-normal.", space_after=0.06),
            MixtureCurve(n=4, caption="one step: four curves mixed"),
            FanV(h=0.5, samples=7, labels=("median", "sample paths")),
            Text("100 paths are drawn; their median is the point forecast.", space_after=0.0),
        ]),
    ],
    footer=[("Any-variate attention takes any number of rows, so covariates and related series go in as they are.",
             SIZE_H["foot"], "bold", INK),
            ("6 blocks · width 384 · 6 heads · patch sizes 8 to 128 · the whole horizon in one pass",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · all covariates · five metals together · "
             "patch 32 · 100 samples · CC-BY-NC-4.0", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.1, 1.0, 1.3, 1.05]
SHARES_2UP = [1.1, 1.0, 1.19, 1.0]   # two-up book figure: rows 1-2 and 3-4

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "moirai11_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
