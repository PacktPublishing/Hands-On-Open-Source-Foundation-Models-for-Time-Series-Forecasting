"""Timestral explainers (209M and 5M share one design)."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, Chip, Chips, Explainer, FanV, PatchToTokenV, QueryTokens,
                       SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "
MATH = r"$\tilde{x}=\mathrm{arcsinh}\left(\dfrac{x-\mu_{\leq k}}{\sigma_{\leq k}}\right)$"


def make(size="209M"):
    big = size == "209M"
    n, d, h, ff = (24, 768, 12, 2048) if big else (8, 192, 3, 512)
    params = "209M" if big else "5.2M"
    return Explainer(
        title=f"How Timestral-{size} makes a forecast",
        subtitle=f"Mistral AI · {params} parameters · causal transformer with forecast queries · zero-shot",
        steps=[
            Step("1. One row per series", "yellow", [
                Text("Targets, past-only and known-ahead covariates. A ? is a value not known yet.",
                     space_after=0.08),
                SeriesRowsStacked([("Target", "temperature", False), ("Known ahead", "day length", True),
                                   ("Past only", "NAO index", False)],
                                  colors=["#2f2f2f", "#2e8a5f", "#9a7b33"], space_after=0.12),
                Text("Covariate rows only steer the forecast; they are never forecast themselves.",
                     size=SIZE_H["small"], color=GREY, space_after=0.0),
            ]),
            Step("2. Scale as you go, squash, patch", "green", [
                Text("Each row is scaled patch by patch with the mean and standard deviation so far, then squashed "
                     "by arcsinh. Known-ahead rows use the whole window.", space_after=0.06),
                Chips([Chip([MATH], math=True)], pad=0.04, space_after=0.1),
                PatchToTokenV(patch_len=32, rows=("value", "missing?"), caption="one patch = 32 steps",
                              token_label=f"one token: {d:,} numbers"),
            ]),
            Step(f"3. {n} blocks; every 4th reads across rows", "orange", [
                Text("After the context tokens come forecast queries: one learned token, repeated once per future "
                     "patch. RMSNorm in front, shortcut around.", space_after=0.06),
                QueryTokens(),
                Chips([Chip(["**Time attention:** causal over the context; queries read the context (RoPE, "
                             "sigmoid gate)"], align="left", edge=ACCENT["blue"], lw=1.3)], pad=0.05,
                      space_after=0.05),
                Chips([Chip([f"**Row attention** ({n // 4} blocks): down the column; covariates never read the "
                             "targets"], align="left", edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
                Chips([Chip([f"**Feed forward:** SwiGLU, {d:,}{NB}→{NB}{ff:,}{NB}→{NB}{d:,}"], align="left")],
                      pad=0.05, space_after=0.0),
            ]),
            Step("4. 1,024 steps in one pass", "blue", [
                Text("A head turns each query into **32 steps × 21 quantiles**: the median plus positive steps, so "
                     "the quantiles never cross.", space_after=0.08),
                FanV(h=0.62),
                Text("The library also forecasts −x and averages the two, and clips at zero a series that was never "
                     "negative.", size=SIZE_H["small"], color=GREY, space_after=0.0),
            ]),
        ],
        footer=[("Learned forecast queries give the whole horizon in one pass, with quantiles that never cross.",
                 SIZE_H["foot"], "bold", INK),
                (f"{n} blocks ({n // 4} read across rows) · width {d:,} · {h} heads · patches of 32 · context up to "
                 "32,768 steps", SIZE_H["small"], "normal", GREY),
                ("In the notebooks: last 2,048 days or full monthly history · all covariates and the five metals "
                 "together · weights: Modified MIT", SIZE_H["small"], "normal", GREY)],
    )


SHARES = [1.0, 1.05, 1.4, 1.0]

if __name__ == "__main__":
    for size, key in [("209M", "timestral"), ("5M", "timestral-5m")]:
        out = Path(f"{key.replace('-', '_')}_explainer")
        fig = draw_explainer_h(make(size), SHARES, path=out)
        print("saved", out, fig.get_size_inches())
