"""Toto-2.0 (22m) explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, AttentionGridC, Chip, Chips, Explainer, FanV, PatchToTokenV,
                       QueryTokens, SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "
MATH = r"$\tilde{x}=\mathrm{arcsinh}\left(\dfrac{x-\mu_{\leq k}}{\sigma_{\leq k}}\right)$"

ex = Explainer(
    title="How Toto-2.0 makes a forecast",
    subtitle="Datadog · 22M parameters · decoder-only transformer, time and row attention · zero-shot",
    steps=[
        Step("1. Targets only, one row each", "yellow", [
            Text("Toto 2.0 reads the targets. Related targets form one group. A ? is a value not known yet.",
                 space_after=0.08),
            SeriesRowsStacked([("Target", "copper", False), ("Target", "nickel", False),
                               ("Target", "zinc", False)],
                              colors=["#2f2f2f", "#2f2f2f", "#2f2f2f"], space_after=0.1),
            Text("Covariates are not supported yet; day length and the NAO index are left out.",
                 size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("2. Scale as you go, squash, patch", "green", [
            Text("Each patch is scaled with the mean and standard deviation of the data up to it, then squashed "
                 "by arcsinh.", space_after=0.06),
            Chips([Chip([MATH], math=True)], pad=0.04, space_after=0.1),
            PatchToTokenV(patch_len=32, rows=("value", "missing?"), caption="one patch = 32 steps",
                          token_label="one token: 512 numbers"),
        ]),
        Step("3. Five time blocks, then a row block", "orange", [
            Text("The future goes in as masked patches M behind the context. RMSNorm in front, shortcut around.",
                 space_after=0.04),
            QueryTokens(q_text="M", right_label="masked future", space_after=0.06),
            AttentionGridC(["copper", "nickel", "zinc"], reg=False, causal=True, n_past=3, n_future=2),
            Chips([Chip(["**Time attention:** causal, along the row (RoPE + xPos)"], align="left",
                        edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Row attention:** the 6th block, down the column across the group"], align="left",
                        edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
            Chips([Chip([f"**Feed forward:** SwiGLU, 512{NB}→{NB}1,368{NB}→{NB}512"], align="left")], pad=0.05,
                  space_after=0.0),
        ]),
        Step("4. Nine quantiles, one pass", "blue", [
            Text("Each token gives **9 quantiles** for the 32 steps of the next patch. With the future masked, the "
                 "whole horizon comes out of **one pass**.", space_after=0.08),
            FanV(h=0.62),
            Text("Undoing arcsinh and the scaling restores the units; the median is the point forecast.",
                 space_after=0.0),
        ]),
    ],
    footer=[("A small decoder that reads related targets together and gives the whole horizon in one pass.",
             SIZE_H["foot"], "bold", INK),
            ("6 blocks (5 time, 1 row) · width 512 · 8 heads · patches of 32 · 9 quantiles",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · five metals together · no covariates · "
             "weights: Apache 2.0", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.05, 1.4, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "toto2_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
