"""TiRex explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, Chip, Chips, Explainer, FanV, PatchToTokenV, SeriesRowsStacked,
                       StateChain, Step, Text, draw_explainer_h)

ex = Explainer(
    title="How TiRex makes a forecast",
    subtitle="NX-AI · 35M parameters · xLSTM (sLSTM) recurrent network · zero-shot",
    steps=[
        Step("1. One series at a time", "yellow", [
            Text("TiRex reads only the target's own past. A ? is a value not known yet.", space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False)], colors=["#2f2f2f"], space_after=0.12),
            Text("Covariates such as day length are not used.", size=SIZE_H["small"], color=GREY, space_after=0.06),
            Text("Five metal prices become five separate forecasts.", size=SIZE_H["small"], color=GREY,
                 space_after=0.0),
        ]),
        Step("2. Scale, then cut into patches", "green", [
            Text("The context is scaled by its own mean μ and standard deviation σ. Gaps stay as missing values.",
                 space_after=0.06),
            Chips([Chip([r"$\tilde{x}=\dfrac{x-\mu}{\sigma}$"], math=True)], pad=0.04, space_after=0.1),
            PatchToTokenV(patch_len=32, rows=("value", "seen?"), caption="one patch = 32 steps",
                          token_label="one token: 512 numbers"),
        ]),
        Step("3. Twelve recurrent blocks", "orange", [
            Text("No attention: an sLSTM cell reads the patches in order and carries a state from one to the next. "
                 "RMSNorm in front, shortcut around.", space_after=0.04),
            StateChain(n_past=5, n_future=0),
            Chips([Chip(["**sLSTM:** 4 heads, exponential input and forget gates, a state that tracks the series"],
                        align="left", edge=ACCENT["green"], lw=1.4)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Feed forward:** gated (SwiGLU), width 1,408"], align="left")], pad=0.05,
                  space_after=0.0),
        ]),
        Step("4. 32 steps per pass", "blue", [
            Text("After a final RMSNorm, an output network turns the last token into **9 quantiles** for each of "
                 "the next **32 steps**.", space_after=0.08),
            FanV(h=0.62),
            Text("For a longer horizon, the forecast patch goes back in as missing values, never as numbers, and "
                 "the next one is predicted. The median is the point forecast.", space_after=0.0),
        ]),
    ],
    footer=[("A recurrent state instead of attention: the cost grows only linearly with the context.",
             SIZE_H["foot"], "bold", INK),
            ("12 blocks · width 512 · 4 heads · patches of 32 in and out · trained on 2,048-step contexts",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · each metal on its own · no covariates · "
             "a 90-day horizon takes 3 passes", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.0, 1.3, 1.1]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tirex_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
