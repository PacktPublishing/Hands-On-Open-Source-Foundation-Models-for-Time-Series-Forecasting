# Copyright (c) 2026 Shovon Sengupta and Tanujit Chakraborty.
# Companion code for "Hands-On Open-Source Foundation Models for Time Series
# Forecasting" (Packt). Licensed under the MIT License.
"""One call per notebook figure: ``figure(key, path=None, show=True)`` draws the step-by-step explainer of a model.

The keys are the ones the Chapter 3 notebooks pass to ``ch03_zeroshot.architectures.figure`` (for example
"chronos-bolt", "lag-llama", "gbdt"). Each figure reads left to right in numbered steps, from the data the model
reads to the forecast it returns. Fonts (Liberation Sans, SIL OFL 1.1) ship in ``fonts/``, so Colab, a Mac and
Linux draw the same picture.
"""
import contextlib
import importlib
import io
import logging

import matplotlib

from .explainer import RC, draw_explainer_h

# notebook key -> (module, size); modules live in specs/, except the Chronos-2 explainer
SPECS = {
    "chronos2": ("chronos2_landscape", None), "chronos-bolt": ("chronos_bolt", None),
    "timesfm25": ("timesfm25", None), "timesfm3": ("timesfm3", None),
    "tirex": ("tirex", None), "tirex2": ("tirex2", None),
    "timestral": ("timestral", "209M"), "timestral-5m": ("timestral", "5M"),
    "ttm": ("ttm", None), "flowstate": ("flowstate", None), "patchtst-fm": ("patchtst_fm", None),
    "t0": ("t0", None), "tabpfn-ts": ("tabpfn_ts", None),
    "moirai11": ("moirai11", None), "moirai-moe": ("moirai_moe", None), "moirai2": ("moirai2", None),
    "toto2": ("toto2", None), "toto1": ("toto1", None),
    "sundial": ("sundial", None), "timer": ("timer", None), "timemoe": ("timemoe", None),
    "kairos": ("kairos", None), "lag-llama": ("lag_llama", None),
    # supervised baselines (Notebook 10 only)
    "dlinear": ("dlinear", None), "nhits": ("nhits", None), "tft": ("tft", None), "tide": ("tide", None),
    "gbdt": ("gbdt", None),
}


def spec(key):
    """The Explainer of ``key`` and the width shares of its panels."""
    module, size = SPECS[key]
    where = f".{module}" if module == "chronos2_landscape" else f".specs.{module}"
    m = importlib.import_module(where, __package__)
    ex = m.make(size) if size else m.ex
    return ex, m.SHARES


@contextlib.contextmanager
def _quiet():
    """Hide matplotlib's font fallback messages (large brackets come from the STIX fallback font)."""
    loggers = [logging.getLogger(n) for n in ("matplotlib.mathtext", "matplotlib.font_manager")]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for lg, lv in zip(loggers, levels):
            lg.setLevel(lv)


def figure(key, path=None, show=True, dpi=200):
    """Draw the explainer of ``key`` left to right; saves ``path`` as .png (``dpi``) and .pdf when given.

    In a notebook (``show=True``) the picture shown is rendered with the figure's own fonts and settings, so it
    is the same image as the saved PNG; the figure itself is returned closed."""
    ex, shares = spec(key)
    with _quiet():
        fig = draw_explainer_h(ex, shares, path=path, show=False, dpi=dpi)
        if show:
            buf = io.BytesIO()
            with matplotlib.rc_context(RC):
                fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
            try:
                from IPython.display import Image, display
                display(Image(data=buf.getvalue()))
            except ImportError:
                pass
    return fig
