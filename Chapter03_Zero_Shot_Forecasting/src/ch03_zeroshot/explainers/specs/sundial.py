"""Sundial (base, 128M) explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, CausalMask, Chip, Chips, Explainer, FanV, FlowSteps,
                       PatchToTokenV, SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "

ex = Explainer(
    title="How Sundial makes a forecast",
    subtitle="Tsinghua University (THUML) · 128M parameters · decoder-only transformer with a flow-matching head · "
             "zero-shot",
    steps=[
        Step("1. One series at a time", "yellow", [
            Text("Sundial reads only the target's own past. A ? is a value not known yet.", space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False)], colors=["#2f2f2f"], space_after=0.12),
            Text("Covariates are not used.", size=SIZE_H["small"], color=GREY, space_after=0.06),
            Text("Five metal prices become five separate forecasts.", size=SIZE_H["small"], color=GREY,
                 space_after=0.0),
        ]),
        Step("2. Scale, then patches of 16", "green", [
            Text("The context is scaled by its mean μ and standard deviation σ. Each patch and its seen flags go "
                 "through a residual MLP.", space_after=0.06),
            Chips([Chip([r"$\tilde{x}=\dfrac{x-\mu}{\sigma}$"], math=True)], pad=0.04, space_after=0.1),
            PatchToTokenV(patch_len=16, rows=("value", "seen?"), caption="one patch = 16 steps",
                          token_label="one token: 768 numbers"),
        ]),
        Step("3. Twelve causal blocks", "orange", [
            Text("Each patch reads only itself and earlier patches. LayerNorm in front, shortcut around.",
                 space_after=0.04),
            CausalMask(n=5, caption="earlier patches only"),
            Chips([Chip(["**Self-attention:** causal, 12 heads, rotary positions (RoPE)"], align="left",
                        edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip([f"**Feed forward:** SwiGLU, 768{NB}→{NB}3,072{NB}→{NB}768"], align="left")], pad=0.05,
                  space_after=0.0),
        ]),
        Step("4. Paths drawn from noise", "blue", [
            Text("The last token steers a small network that turns random noise into a **720-step path** in 50 "
                 "small steps (flow matching).", space_after=0.06),
            FlowSteps(),
            FanV(h=0.5, samples=7, labels=("median", "sample paths")),
            Text("A fresh noise draw gives another path; the median of the paths is the point forecast.",
                 space_after=0.0),
        ]),
    ],
    footer=[("A flow-matching head draws whole forecast paths instead of a fixed set of quantiles.",
             SIZE_H["foot"], "bold", INK),
            ("12 blocks · width 768 · 12 heads · patches of 16 · up to 720 steps per pass · 50 flow steps",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · each metal on its own · no covariates · "
             "20 sample paths", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.0, 1.15, 1.2]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sundial_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
