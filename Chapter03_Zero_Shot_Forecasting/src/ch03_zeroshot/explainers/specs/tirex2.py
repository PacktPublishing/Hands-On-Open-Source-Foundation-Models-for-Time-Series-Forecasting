"""TiRex-2 explainer."""
import sys
from pathlib import Path

from ..explainer import (ACCENT, GREY, INK, SIZE_H, AttentionGridC, Chip, Chips, Explainer, FanV, PatchToTokenV,
                       SeriesRowsStacked, Step, Text, draw_explainer_h)

ex = Explainer(
    title="How TiRex-2 makes a forecast",
    subtitle="NX-AI · 83M parameters · xLSTM along time, attention across series · zero-shot",
    steps=[
        Step("1. One row per series", "yellow", [
            Text("Targets, past-only and known-ahead covariates. A ? is a value not known yet.", space_after=0.08),
            SeriesRowsStacked([("Target", "temperature", False), ("Known ahead", "day length", True),
                               ("Past only", "NAO index", False)],
                              colors=["#2f2f2f", "#2e8a5f", "#9a7b33"], space_after=0.12),
            Text("Five metal prices go in together as five target rows.", size=SIZE_H["small"], color=GREY,
                 space_after=0.0),
        ]),
        Step("2. Scale, squash, then patch", "green", [
            Text("Each row is scaled by its own mean μ and standard deviation σ, then squashed by arcsinh; "
                 "0/1 rows are left as they are.", space_after=0.06),
            Chips([Chip([r"$\tilde{x}=\mathrm{arcsinh}\left(\dfrac{x-\mu}{\sigma}\right)$"], math=True)], pad=0.04,
                  space_after=0.1),
            PatchToTokenV(patch_len=32, rows=("value", "seen?"), caption="one patch = 32 steps",
                          token_label="one token: 512 numbers", space_after=0.06),
            Text("A target with a strong straight-line trend is forecast as differences.", size=SIZE_H["small"],
                 color=GREY, space_after=0.0),
        ]),
        Step("3. Twelve blocks: time, then rows", "orange", [
            Text("Each block mixes along time, then across rows. Every part has a norm in front and a shortcut "
                 "around it.", space_after=0.04),
            AttentionGridC(["temperature", "day length", "NAO index"], reg=False, causal=True, fill_rows=[1]),
            Chips([Chip(["**Time mixer:** xLSTM along the row (mLSTM and sLSTM in turn); known-ahead rows also "
                         "run backwards"], align="left", edge=ACCENT["blue"], lw=1.3)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Row mixer:** attention down the column; targets read covariates, never the reverse"],
                        align="left", edge=ACCENT["red"], lw=1.5)], pad=0.05, space_after=0.05),
            Chips([Chip(["**Feed forward** after each mixer, token by token"], align="left")], pad=0.05,
                  space_after=0.0),
        ]),
        Step("4. 320 steps, one pass", "blue", [
            Text("Ten future patches go in as missing values. After a final norm, a head turns each into **32 "
                 "steps × 9 quantiles**.", space_after=0.08),
            FanV(h=0.62),
            Text("A fixed calibration rescales the bands, the scaling is undone, and the median is the point "
                 "forecast.", space_after=0.0),
        ]),
    ],
    footer=[("Recurrence along time, attention across rows: covariates steer the targets, never the reverse.",
             SIZE_H["foot"], "bold", INK),
            ("12 blocks · width 512 · 4 heads · patches of 32 · context 2,048 steps · up to 320 steps per pass",
             SIZE_H["small"], "normal", GREY),
            ("In the notebooks: last 2,048 days or full monthly history · all covariates and the five metals "
             "together · Apache-2.0", SIZE_H["small"], "normal", GREY)],
)

SHARES = [1.05, 1.0, 1.35, 1.0]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tirex2_explainer")
    fig = draw_explainer_h(ex, SHARES, path=out)
    print("saved", out, fig.get_size_inches())
