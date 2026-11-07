"""Chronos-2 explainer, left to right (landscape), same content as the vertical sample."""
import sys
from pathlib import Path

from .explainer import (ACCENT, GREY, INK, SIZE_H, AttentionGrid, Chip, Chips, Explainer, FanV, PatchToTokenV,
                       SeriesRowsStacked, Stack, Step, Text, TokenSequence, AttentionGridC, draw_explainer_h)

MATH_ASINH = r"$\tilde{x}=\mathrm{arcsinh}\left(\dfrac{x-\mu}{\sigma}\right)$"
NB = " "   # keeps short expressions on one line

ex = Explainer(
    title="How Chronos-2 makes a forecast",
    subtitle="Amazon · 120M parameters · encoder-only transformer · zero-shot",
    steps=[
        Step("1. One row per series", "yellow", [
            Text("The target and its covariates form one group. A ? is a value not known yet.", space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False),
                               ("Known ahead", "day length", True),
                               ("Past only", "NAO index", False)],
                              colors=["#2f2f2f", "#2e8a5f", "#9a7b33"], space_after=0.12),
            Text("Several targets, such as five metal prices, can share one group.", size=SIZE_H["small"],
                 color=GREY, space_after=0.0),
        ]),
        Step("2. Scale, then cut into patches", "green", [
            Text("Each row is scaled by its own past mean μ and standard deviation σ, then squashed by "
                 "arcsinh.", space_after=0.06),
            Chips([Chip([MATH_ASINH], math=True)], pad=0.04, space_after=0.1),
            PatchToTokenV(caption="one patch = 16 steps", token_label="one token: 768 numbers"),
        ]),
        Step("3. Twelve blocks mix the tokens", "orange", [
            Text("Each layer starts with RMSNorm and has a shortcut around it. The orange REG token marks where "
                 "the future starts.", space_after=0.04),
            AttentionGridC(["temperature", "day length", "NAO index"]),
            Chips([Chip(["**Time attention:** along the row, all past and future patches (RoPE positions)"],
                        align="left", edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Group attention:** down the column, the other rows at the same patch"],
                        align="left", edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
            Chips([Chip([f"**Feed forward:** each token on its own, 768{NB}→{NB}3,072{NB}→{NB}768, ReLU"],
                        align="left")], pad=0.05, space_after=0.0),
        ]),
        Step("4. All quantiles in one pass", "blue", [
            Text("After a final RMSNorm, an output network turns each future token into **21 quantiles** for "
                 "each of its 16 steps.", space_after=0.08),
            FanV(h=0.72),
            Text(f"x{NB}={NB}μ{NB}+{NB}σ{NB}·{NB}sinh(x̃) restores the units; the median is the point forecast.",
                 space_after=0.0),
        ]),
    ],
    footer=[("Group attention lets covariates and related series shape the forecast.", SIZE_H["foot"], "bold", INK),
            ("12 blocks · width 768 · 12 heads · context up to 8,192 steps · up to 1,024 steps per pass",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · one group per forecast · quantiles 0.1 "
             "to 0.9 kept", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.0, 1.45, 1.0]
SHARES_2UP = [1.05, 0.95, 1.45, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "chronos2_explainer_landscape")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
