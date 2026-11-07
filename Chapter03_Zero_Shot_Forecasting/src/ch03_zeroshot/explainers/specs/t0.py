"""t0-gamma explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, AttentionGridC, Chip, Chips, Explainer, FanV, PatchToTokenV,
                       SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "
MATH = r"$\tilde{x}=\mathrm{arcsinh}\left(\dfrac{x-\mu_{\leq t}}{\sigma_{\leq t}}\right)$"

ex = Explainer(
    title="How t0 makes a forecast",
    subtitle="The Forecasting Company · t0-gamma, 256M parameters · time + variate attention · zero-shot",
    steps=[
        Step("1. Rows with roles", "yellow", [
            Text("Each row is tagged as target, past-only or known-ahead. A ? is a value not known yet.",
                 space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False), ("Known ahead", "day length", True),
                               ("Past only", "NAO index", False)],
                              colors=["#2f2f2f", "#2e8a5f", "#9a7b33"], space_after=0.12),
            Text("Over the horizon, target and past-only values are withheld; known-ahead rows carry theirs.",
                 size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("2. Scale as you go, squash, patch", "green", [
            Text("Each step is scaled with statistics up to it, then squashed by arcsinh. Known-ahead rows use the "
                 "whole window.", space_after=0.06),
            Chips([Chip([MATH], math=True)], pad=0.04, space_after=0.1),
            PatchToTokenV(patch_len=32, rows=("value", "time", "seen?"), caption="one patch = 32 steps",
                          token_label="one token: 1,024 numbers", space_after=0.06),
            Text("A role embedding tells the model which kind of row a token belongs to.", size=SIZE_H["small"],
                 color=GREY, space_after=0.0),
        ]),
        Step("3. 24 layers: two along time, one across rows", "orange", [
            Text("RMSNorm in front of every layer, a shortcut around it.", space_after=0.04),
            AttentionGridC(["temperature", "day length", "NAO index"], reg=False, causal=True, fill_rows=[1]),
            Chips([Chip(["**Time attention** (16 layers): causal; known-ahead rows read both ways (RoPE with decay)"],
                        align="left", edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Variate attention** (8 layers): down the column; known-ahead rows do not read back"],
                        align="left", edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
            Chips([Chip([f"**Feed forward:** SwiGLU, 1,024{NB}→{NB}2,048{NB}→{NB}1,024"], align="left")],
                  pad=0.05, space_after=0.0),
        ]),
        Step("4. 21 quantiles that never cross", "blue", [
            Text("After a final RMSNorm, each future token becomes **32 steps × 21 quantiles**, built as positive "
                 "steps so they never cross.", space_after=0.08),
            FanV(h=0.62),
            Text("Up to 1,024 steps in one pass. The median is the point forecast.", space_after=0.0),
        ]),
    ],
    footer=[("Role-aware masks let covariates inform the target without leaking the future backwards.",
             SIZE_H["foot"], "bold", INK),
            ("24 layers (16 time, 8 variate) · width 1,024 · 8 heads · patches of 32 · up to 1,024 steps per pass",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · all covariates and the five metals "
             "together · each length run on its own", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.05, 1.05, 1.35, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "t0_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
