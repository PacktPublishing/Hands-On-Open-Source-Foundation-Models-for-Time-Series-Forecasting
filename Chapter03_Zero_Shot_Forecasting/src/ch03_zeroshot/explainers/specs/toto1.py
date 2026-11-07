"""Toto-1.0 (Open-Base) explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, AttentionGridC, Chip, Chips, Explainer, FanV, MixtureCurve,
                       PatchToTokenV, SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "
MATH = r"$\tilde{x}=\dfrac{x-\mu_{\leq k}}{\sigma_{\leq k}}$"

ex = Explainer(
    title="How Toto-1.0 makes a forecast",
    subtitle="Datadog · 151M parameters · decoder-only transformer, time and row attention · zero-shot",
    steps=[
        Step("1. One row per series", "yellow", [
            Text("Targets form one group of rows; known-ahead covariates join as extra rows. A ? is a value not "
                 "known yet.", space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False), ("Known ahead", "day length", True)],
                              colors=["#2f2f2f", "#2e8a5f"], space_after=0.1),
            Text("Past-only covariates such as the NAO index are not used.", size=SIZE_H["small"], color=GREY,
                 space_after=0.0),
        ]),
        Step("2. Scale as you go, patches of 64", "green", [
            Text("Each patch is scaled with the mean and standard deviation of the data up to it, so no future "
                 "value leaks in.", space_after=0.06),
            Chips([Chip([MATH], math=True)], pad=0.04, space_after=0.1),
            PatchToTokenV(patch_len=64, rows=("value",), caption="one patch = 64 steps", mlp_label="linear layer",
                          token_label="one token: 768 numbers"),
        ]),
        Step("3. Eleven time blocks, then a row block", "orange", [
            Text("The last context token (black) predicts the next patch. RMSNorm in front, shortcut around.",
                 space_after=0.04),
            AttentionGridC(["temperature", "day length"], reg=False, causal=True, n_past=4, n_future=2,
                           focus_col=3, fill_rows=[1], row_colors=[GREY, ACCENT["green"]]),
            Chips([Chip(["**Time attention:** causal, along the row (RoPE + xPos); 11 blocks"], align="left",
                        edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Row attention:** the 12th block, down the column across the group"], align="left",
                        edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
            Chips([Chip([f"**Feed forward:** SwiGLU, 768{NB}→{NB}3,072{NB}→{NB}768"], align="left")], pad=0.05,
                  space_after=0.0),
        ]),
        Step("4. Sample 64 steps at a time", "blue", [
            Text("The last token gives a **mix of 24 Student-t curves** for each of the next 64 steps.",
                 space_after=0.04),
            MixtureCurve(h=0.45, n=3, caption="one step: a Student-t mix"),
            Text("A sample is drawn and appended, with the real values for known-ahead rows; then the next 64.",
                 space_after=0.06),
            FanV(h=0.5, samples=7, labels=("median", "sample paths")),
            Text("So day length ahead can only steer the steps after the first 64.", size=SIZE_H["small"],
                 color=GREY, space_after=0.0),
        ]),
    ],
    footer=[("Eleven time blocks for every row block keep attention cheap while related series still meet.",
             SIZE_H["foot"], "bold", INK),
            ("12 blocks (11 time, 1 row) · width 768 · 12 heads · patches of 64 · 24 Student-t components",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · day length and calendar terms for D2 · "
             "32 sample paths, median as the point", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.0, 1.4, 1.05]
SHARES_2UP = [1.0, 1.0, 1.45, 1.0]   # two-up book figure: rows 1-2 and 3-4

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "toto1_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
