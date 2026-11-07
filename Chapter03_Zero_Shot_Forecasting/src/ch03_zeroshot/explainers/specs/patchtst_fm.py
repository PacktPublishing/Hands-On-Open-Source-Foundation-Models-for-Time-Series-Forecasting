"""PatchTST-FM r2 explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, Chip, Chips, Explainer, FanV, OverlapPatches, SeriesRowsStacked,
                       Step, Text, draw_explainer_h)

NB = " "

ex = Explainer(
    title="How PatchTST-FM makes a forecast",
    subtitle="IBM · 385M parameters · Conformer encoder over overlapping patches · zero-shot",
    steps=[
        Step("1. One series at a time", "yellow", [
            Text("PatchTST-FM reads only the target's own past. A ? is a value not known yet.", space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False)], colors=["#2f2f2f"], space_after=0.12),
            Text("Covariates are not used; five metal prices become five separate forecasts.", size=SIZE_H["small"],
                 color=GREY, space_after=0.0),
        ]),
        Step("2. Overlapping patches", "green", [
            Text("The context is scaled by its own mean μ and standard deviation σ. Patches of 16 steps start "
                 "every 8 steps, so neighbours overlap by half.", space_after=0.08),
            Chips([Chip([r"$\tilde{x}=\dfrac{x-\mu}{\sigma}$"], math=True)], pad=0.04, space_after=0.1),
            OverlapPatches(caption="dashed: masked future"),
            Text("Values and mask go through a residual MLP; a learned position embedding is added.",
                 size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("3. Thirty Conformer blocks", "orange", [
            Text("Forecasting means filling in the masked future patches. Every part has a LayerNorm in front and a "
                 "shortcut around it.", space_after=0.06),
            Chips([Chip(["**½ feed forward** (GELU)"], align="left")], pad=0.05, space_after=0.05),
            Chips([Chip(["**Self-attention:** 16 heads, every patch reads every other, past and future"],
                        align="left", edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Convolution:** depthwise, kernel 5 or 3, smooths across neighbouring patches"],
                        align="left", edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
            Chips([Chip(["**½ feed forward** (GELU)"], align="left")], pad=0.05, space_after=0.0),
        ]),
        Step("4. 99 quantiles, blended", "blue", [
            Text("A head turns each future patch into **99 quantiles** (0.01 to 0.99) per step. Overlapping "
                 "patches are blended with a Hamming window.", space_after=0.08),
            FanV(h=0.62),
            Text("The point forecast is the mean: the area under the quantile curve, not the median.",
                 space_after=0.0),
        ]),
    ],
    footer=[("A wide, bidirectional encoder fills in the future like a gap in the data, in one pass.",
             SIZE_H["foot"], "bold", INK),
            ("30 blocks · width 1,024 · 16 heads · patches of 16 every 8 steps · context up to 8,192 steps",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · each metal on its own · no covariates · "
             "quantiles 0.1 to 0.9 kept", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.05, 1.25, 1.05]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "patchtst_fm_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
