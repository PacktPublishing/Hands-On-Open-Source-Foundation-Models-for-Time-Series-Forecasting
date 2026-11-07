"""Moirai-MoE (small) explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, BlockAttention, Chip, Chips, ExpertRouter, Explainer, FanV,
                       FlatSequence, PatchToTokenV, SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "

ex = Explainer(
    title="How Moirai-MoE makes a forecast",
    subtitle="Salesforce · 117M parameters, 11M used per token (small) · decoder-only mixture of experts · zero-shot",
    steps=[
        Step("1. All rows in one sequence", "yellow", [
            Text("Targets and covariates are laid end to end as one long sequence. A ? is a value not known yet.",
                 space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False), ("Known ahead", "day length", True),
                               ("Past only", "NAO index", False)],
                              colors=["#2f2f2f", "#2e8a5f", "#9a7b33"], space_after=0.1),
            FlatSequence(future_text="?"),
            Text("Any number of rows fits, so the five metals share one sequence.", size=SIZE_H["small"],
                 color=GREY, space_after=0.0),
        ]),
        Step("2. Scale, then patches of 16", "green", [
            Text("Each row is scaled by the mean μ and standard deviation σ of its own past. One patch size for "
                 "every frequency.", space_after=0.06),
            Chips([Chip([r"$\tilde{x}=\dfrac{x-\mu}{\sigma}$"], math=True)], pad=0.04, space_after=0.1),
            PatchToTokenV(patch_len=16, rows=("value",), caption="one patch = 16 steps",
                          token_label="one token: 384 numbers"),
        ]),
        Step("3. Six causal blocks with experts", "orange", [
            Text("Each token reads the tokens up to its own time, in every row. RMSNorm in front, shortcut around.",
                 space_after=0.06),
            BlockAttention(causal=True),
            Chips([Chip(["**Attention:** RoPE in time, plus a same-row / other-row bias"], align="left",
                        edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.06),
            ExpertRouter(),
            Chips([Chip(["**Experts:** the gate scores each token against 32 cluster centres from a pre-trained "
                         "Moirai; each expert is a SwiGLU feed forward (512)"],
                        align="left", edge=ACCENT["orange"], lw=1.4)], pad=0.05, space_after=0.0),
        ]),
        Step("4. Sample patch by patch", "blue", [
            Text("Each token gives a **mix of four distributions** for every step of the next patch. A sample is "
                 "drawn, appended, and the model runs again.", space_after=0.08),
            FanV(h=0.62, samples=7, labels=("median", "sample paths")),
            Text("100 paths; their median is the point forecast.", space_after=0.0),
        ]),
    ],
    footer=[("Experts specialise on kinds of tokens, so one patch size serves every frequency.",
             SIZE_H["foot"], "bold", INK),
            ("6 blocks · width 384 · 6 heads · 32 experts, 2 per token · patches of 16 · mixture: Student-t, "
             "normal, negative binomial, log-normal", SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · all covariates · five metals together · "
             "100 samples · 90 days = 6 rounds", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.0, 1.5, 1.0]
SHARES_2UP = [1.05, 0.95, 1.5, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "moirai_moe_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
