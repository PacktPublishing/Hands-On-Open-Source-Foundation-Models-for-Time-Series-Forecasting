"""TTM-r2 explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, Chip, Chips, Explainer, FanV, MixerGrid, PatchBar,
                       SeriesRowsStacked, Step, Text, draw_explainer_h)

ex = Explainer(
    title="How TTM-r2 makes a forecast",
    subtitle="IBM · 0.8M parameters · MLP mixer, no attention · zero-shot",
    steps=[
        Step("1. One series, 512 steps", "yellow", [
            Text("TTM reads exactly the last 512 steps of one series. A ? is a value not known yet.",
                 space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False)], colors=["#2f2f2f"], space_after=0.12),
            Text("A shorter history is padded and marked as unseen.", size=SIZE_H["small"], color=GREY,
                 space_after=0.06),
            Text("Covariates are not used zero-shot; each metal is forecast on its own.", size=SIZE_H["small"],
                 color=GREY, space_after=0.0),
        ]),
        Step("2. Scale, then 8 patches", "green", [
            Text("The context is scaled by its own mean μ and standard deviation σ and cut into 8 patches.",
                 space_after=0.06),
            Chips([Chip([r"$\tilde{x}=\dfrac{x-\mu}{\sigma}$"], math=True)], pad=0.04, space_after=0.08),
            PatchBar(token_label="9 tokens of 192 numbers", space_after=0.06),
            Text("The orange token in front marks the sampling frequency.", size=SIZE_H["small"], color=GREY,
                 space_after=0.0),
        ]),
        Step("3. Mixers instead of attention", "orange", [
            Text("Small MLPs mix the tokens along time, then across features. Patches are merged and split over 3 "
                 "levels.", space_after=0.04),
            MixerGrid(),
            Chips([Chip(["**Time mixing:** one MLP across the patches, for each feature"], align="left",
                        edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Feature mixing:** one MLP across the 192 numbers of each patch"], align="left",
                        edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Gate:** a softmax weighting of features, not self-attention"], align="left")],
                  pad=0.05, space_after=0.0),
        ]),
        Step("4. A point forecast", "blue", [
            Text("A small mixer decoder and a linear head give a **point forecast** for 48 or 96 steps; the first H "
                 "are kept.", space_after=0.08),
            FanV(h=0.62, bands=False, labels=("point forecast",)),
            Text("IBM ships one branch per context and horizon, such as 512-48; the library picks one that "
                 "covers H. No quantiles.", space_after=0.0),
        ]),
    ],
    footer=[("Tiny and attention-free: TTM forecasts in milliseconds on a CPU, but gives point forecasts only.",
             SIZE_H["foot"], "bold", INK),
            ("2 mixer layers per level · 3 levels · width 192 · decoder width 128 · patches of 64 · 512-step context",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: 512-step branches picked per horizon · last 512 steps · each metal on its own · no "
             "covariates · no intervals", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.0, 1.05, 1.3, 1.05]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "ttm_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
