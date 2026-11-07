"""Chronos-Bolt explainer (landscape)."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, Chip, Chips, EncDec, Explainer, FanV, PatchToTokenV,
                       SeriesRowsStacked, Step, Text, draw_explainer_h)

NB = " "

ex = Explainer(
    title="How Chronos-Bolt makes a forecast",
    subtitle="Amazon · 48M parameters · T5 encoder-decoder · zero-shot",
    steps=[
        Step("1. One series at a time", "yellow", [
            Text("Chronos-Bolt reads only the target's own past. A ? is a value not known yet.", space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False)], colors=["#2f2f2f"], space_after=0.12),
            Text("Covariates such as day length are not used.", size=SIZE_H["small"], color=GREY, space_after=0.06),
            Text("Five metal prices become five separate forecasts.", size=SIZE_H["small"], color=GREY,
                 space_after=0.0),
        ]),
        Step("2. Scale, then cut into patches", "green", [
            Text("The context is scaled by its own mean μ and standard deviation σ.", space_after=0.06),
            Chips([Chip([r"$\tilde{x}=\dfrac{x-\mu}{\sigma}$"], math=True)], pad=0.04, space_after=0.1),
            PatchToTokenV(rows=("value", "seen?"), caption="one patch = 16 steps",
                          token_label="one token: 512 numbers"),
        ]),
        Step("3. Encoder reads, decoder writes", "orange", [
            Text("Each layer starts with RMSNorm and has a shortcut around it. A REG token closes the context.",
                 space_after=0.04),
            EncDec(),
            Chips([Chip(["**Encoder, 6 blocks:** self-attention over all context patches (T5 relative positions)"],
                        align="left", edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Decoder, 6 blocks:** one start token reads the encoder through cross-attention"],
                        align="left", edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
            Chips([Chip([f"**Feed forward:** 512{NB}→{NB}2,048{NB}→{NB}512, ReLU"], align="left")], pad=0.05,
                  space_after=0.0),
        ]),
        Step("4. 64 steps at once", "blue", [
            Text("An output network turns the decoder token into **9 quantiles** (0.1 to 0.9) for each of **64 "
                 "steps**.", space_after=0.08),
            FanV(h=0.62),
            Text(f"x{NB}={NB}μ{NB}+{NB}σ{NB}·{NB}x̃ restores the units; the median is the point forecast.",
                 space_after=0.06),
            Text("Past 64 steps, each of the 9 quantile paths is fed back in and the 81 results are cut back to 9 "
                 "quantiles.", size=SIZE_H["small"], color=GREY, space_after=0.0),
        ]),
    ],
    footer=[("A single decoder pass writes 64 steps, so Chronos-Bolt is one of the fastest models here.",
             SIZE_H["foot"], "bold", INK),
            ("6 encoder + 6 decoder blocks · width 512 · 8 heads · context up to 2,048 steps · 64 steps per pass",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: chronos-bolt-small · last 2,048 steps · no covariates · each target channel on its "
             "own · 90-day horizon takes two passes", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 0.95, 1.4, 1.05]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "chronos_bolt_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
