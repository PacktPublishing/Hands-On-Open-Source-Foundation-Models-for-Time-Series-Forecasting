"""Moirai-2.0 (small) explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, BranchFan, CausalMask, Chip, Chips, Explainer, FanV,
                       PatchToTokenV, SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "

ex = Explainer(
    title="How Moirai-2.0 makes a forecast",
    subtitle="Salesforce · 11M parameters (small) · decoder-only transformer · zero-shot",
    steps=[
        Step("1. One series at a time", "yellow", [
            Text("Moirai 2.0 reads only the target's own past. A ? is a value not known yet.", space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False)], colors=["#2f2f2f"], space_after=0.12),
            Text("It was trained on single series, so covariates are not passed.", size=SIZE_H["small"],
                 color=GREY, space_after=0.06),
            Text("Five metal prices become five separate forecasts.", size=SIZE_H["small"], color=GREY,
                 space_after=0.0),
        ]),
        Step("2. Scale, then patches of 16", "green", [
            Text("The context is scaled by its mean μ and standard deviation σ. Each patch keeps a flag for "
                 "missing values.", space_after=0.06),
            Chips([Chip([r"$\tilde{x}=\dfrac{x-\mu}{\sigma}$"], math=True)], pad=0.04, space_after=0.1),
            PatchToTokenV(patch_len=16, rows=("value", "missing?"), caption="one patch = 16 steps",
                          token_label="one token: 384 numbers"),
        ]),
        Step("3. Six causal blocks", "orange", [
            Text("Each patch reads only itself and earlier patches. RMSNorm in front, shortcut around.",
                 space_after=0.04),
            CausalMask(n=5, caption="earlier patches only"),
            Chips([Chip(["**Self-attention:** causal, 6 heads, rotary positions (RoPE)"], align="left",
                        edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip([f"**Feed forward:** SwiGLU, 384{NB}→{NB}1,024{NB}→{NB}384"], align="left")], pad=0.05,
                  space_after=0.0),
        ]),
        Step("4. Four patches per token", "blue", [
            Text("The last token gives **9 quantiles** for each step of the next **4 patches (64 steps)**.",
                 space_after=0.06),
            FanV(h=0.5),
            Text("Longer horizons: all 9 quantile paths go back in, and 9 quantiles are taken from the 81 "
                 "results.", space_after=0.04),
            BranchFan(h=0.5),
        ]),
    ],
    footer=[("A small decoder trained on quantiles that keeps the spread when it rolls forward.",
             SIZE_H["foot"], "bold", INK),
            ("6 blocks · width 384 · 6 heads · patches of 16 · 4 patches × 9 quantiles per token",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 1,680 days or full monthly history · no covariates · median as the point · "
             "90 days = 2 passes · CC-BY-NC-4.0", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.0, 1.15, 1.15]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "moirai2_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
