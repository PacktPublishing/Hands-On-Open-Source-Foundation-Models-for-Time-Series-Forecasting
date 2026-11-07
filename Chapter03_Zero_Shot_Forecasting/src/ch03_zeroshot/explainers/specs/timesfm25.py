"""TimesFM-2.5 explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, CausalMask, Chip, Chips, Explainer, FanV, PatchToTokenV,
                       SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "

ex = Explainer(
    title="How TimesFM-2.5 makes a forecast",
    subtitle="Google · 231M parameters · decoder-only transformer · zero-shot",
    steps=[
        Step("1. One series, read like text", "yellow", [
            Text("The network reads one series at a time. A ? is a value not known yet.", space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False), ("Known ahead", "day length", True)],
                              colors=["#2f2f2f", "#2e8a5f"], space_after=0.1),
            Text("Known-ahead covariates do not enter the network. A linear fit on them (XReg) runs beside it, and "
                 "TimesFM forecasts what the fit leaves over.", size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
        Step("2. Scale as you go, then patch", "green", [
            Text("The context is standardised once. Inside the model, each patch is scaled again with the mean and "
                 "standard deviation of the values up to it.", space_after=0.06),
            Chips([Chip([r"$\tilde{x}_k=\dfrac{x_k-\mu_{\leq k}}{\sigma_{\leq k}}$"], math=True)], pad=0.04,
                  space_after=0.1),
            PatchToTokenV(patch_len=32, rows=("value", "seen?"), caption="one patch = 32 steps",
                          token_label="one token: 1,280 numbers"),
        ]),
        Step("3. Twenty causal blocks", "orange", [
            Text("Each patch reads only itself and earlier patches. RMSNorm sits before and after every layer, "
                 "with a shortcut around it.", space_after=0.04),
            CausalMask(caption="the last patch reads them all"),
            Chips([Chip(["**Causal self-attention:** 16 heads, rotary positions (RoPE), normalised queries and "
                         "keys"], align="left", edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Feed forward:** each token on its own, width 1,280, SiLU"], align="left")], pad=0.05,
                  space_after=0.0),
        ]),
        Step("4. 128 steps from the last token", "blue", [
            Text("The last token's output gives the next **128 steps**. A second head adds **9 quantiles** for up "
                 "to 1,024 steps.", space_after=0.08),
            FanV(h=0.62),
            Text("The scaling is undone and the median is the point forecast.", space_after=0.06),
            Text("The library also forecasts −x and averages the two, and clips at zero a series that was never "
                 "negative.", size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
    ],
    footer=[("A causal model with 128-step outputs: one pass covers every horizon in the notebooks.",
             SIZE_H["foot"], "bold", INK),
            ("20 blocks · width 1,280 · 16 heads · patches of 32 in, 128 out · context up to 16,384 steps",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · each metal on its own · XReg on D2's "
             "calendar covariates · no past-only covariates", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.05, 1.0, 1.25, 1.05]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "timesfm25_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
