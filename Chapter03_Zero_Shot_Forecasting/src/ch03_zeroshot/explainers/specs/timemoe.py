"""Time-MoE (TimeMoE-50M) explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, CausalMask, Chip, Chips, ExpertRouter, Explainer, FanV, PatchBar,
                       SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "

ex = Explainer(
    title="How Time-MoE makes a forecast",
    subtitle="Shi et al. · 113M parameters, 50M used per step (TimeMoE-50M) · decoder-only mixture of experts · "
             "zero-shot",
    steps=[
        Step("1. One series at a time", "yellow", [
            Text("Time-MoE reads only the target's own past. A ? is a value not known yet.", space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False)], colors=["#2f2f2f"], space_after=0.12),
            Text("No covariates; each metal on its own.", size=SIZE_H["small"], color=GREY, space_after=0.06),
            Text("Context plus horizon must stay within 4,096 steps.", size=SIZE_H["small"], color=GREY,
                 space_after=0.0),
        ]),
        Step("2. Every step a token", "green", [
            Text("The model does not scale its input, so our code scales the context by its mean μ and standard "
                 "deviation σ first.", space_after=0.06),
            Chips([Chip([r"$\tilde{x}=\dfrac{x-\mu}{\sigma}$"], math=True)], pad=0.04, space_after=0.1),
            PatchBar(n=8, bar_label="one value per token", arrow_label="gated layer", prefix=None,
                     token_label="8 tokens of 384 numbers"),
        ]),
        Step("3. Twelve causal blocks with experts", "orange", [
            Text("Each token reads itself and earlier tokens. RMSNorm in front, shortcut around.", space_after=0.04),
            CausalMask(n=5, caption="earlier steps only"),
            Chips([Chip(["**Self-attention:** causal, 12 heads, rotary positions (RoPE)"], align="left",
                        edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.06),
            ExpertRouter(n_show=6, picked=(1, 3), caption="2 of 8 experts, plus a shared one"),
            Chips([Chip(["**Experts:** a softmax gate picks 2 of 8 SwiGLU experts (768); a shared expert (1,536) "
                         "always runs, weighted by a sigmoid gate"], align="left", edge=ACCENT["orange"], lw=1.4)],
                  pad=0.05, space_after=0.0),
        ]),
        Step("4. Heads for 1, 8, 32 or 64 steps", "blue", [
            Text("Each pass uses the **longest head that fits** the steps still missing, appends the result and "
                 "runs again.", space_after=0.08),
            FanV(h=0.62, bands=False, labels=("point forecast",)),
            Text(f"90 days = 64{NB}+{NB}8{NB}+{NB}8{NB}+{NB}8{NB}+{NB}1{NB}+{NB}1: six passes. A point forecast "
                 "only.", space_after=0.0),
        ]),
    ],
    footer=[("Sparse experts give a large model at the cost of a small one, with every step read as its own token.",
             SIZE_H["foot"], "bold", INK),
            ("12 blocks · width 384 · 12 heads · 8 experts, 2 per token, plus 1 shared · output heads of 1, 8, 32, 64",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · each metal on its own · no covariates · "
             "no intervals", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.0, 1.45, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "timemoe_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
