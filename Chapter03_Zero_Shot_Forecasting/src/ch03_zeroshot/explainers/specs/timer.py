"""Timer (base, 84M) explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, CausalMask, Chip, Chips, Explainer, FanV, PatchBar,
                       SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "

ex = Explainer(
    title="How Timer makes a forecast",
    subtitle="Tsinghua University (THUML) · 84M parameters · GPT-style decoder, next-patch prediction · zero-shot",
    steps=[
        Step("1. One series at a time", "yellow", [
            Text("Timer reads only the target's own past. A ? is a value not known yet.", space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False)], colors=["#2f2f2f"], space_after=0.12),
            Text("It needs at least 96 steps; older steps are dropped so the length is a multiple of 96.",
                 size=SIZE_H["small"], color=GREY, space_after=0.06),
            Text("No covariates; each metal on its own.", size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("2. Scale, then patches of 96", "green", [
            Text("The context is scaled by its mean μ and standard deviation σ, then cut into patches of 96 "
                 "steps.", space_after=0.06),
            Chips([Chip([r"$\tilde{x}=\dfrac{x-\mu}{\sigma}$"], math=True)], pad=0.04, space_after=0.1),
            PatchBar(n=6, bar_label="context = 6 patches of 96", arrow_label="linear layer", prefix=None,
                     token_label="6 tokens of 1,024 numbers"),
        ]),
        Step("3. Eight causal blocks", "orange", [
            Text("Each patch reads only itself and earlier patches, as in a GPT language model. LayerNorm in "
                 "front, shortcut around.", space_after=0.04),
            CausalMask(n=5, caption="earlier patches only"),
            Chips([Chip(["**Self-attention:** causal, 8 heads, rotary positions (RoPE)"], align="left",
                        edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip([f"**Feed forward:** SwiGLU, 1,024{NB}→{NB}2,048{NB}→{NB}1,024"], align="left")],
                  pad=0.05, space_after=0.0),
        ]),
        Step("4. The next 96 steps", "blue", [
            Text("A linear head turns the last token into the **next 96 values**. A longer horizon appends them and "
                 "runs again.", space_after=0.08),
            FanV(h=0.62, bands=False, labels=("point forecast",)),
            Text("Timer gives a point forecast only, with no intervals.", space_after=0.0),
        ]),
    ],
    footer=[("Forecasting as next-token prediction: the GPT recipe applied to patches of a series.",
             SIZE_H["foot"], "bold", INK),
            ("8 blocks · width 1,024 · 8 heads · patches of 96 in and out", SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · each metal on its own · no covariates · "
             "every horizon in one pass", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.05, 1.15, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "timer_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
