"""Readable figures and well-formed result tables.

* every architecture figure (left to right) keeps its text inside its boxes, keeps labels, maths and captions off
  the boxes, has no overlapping text and stays inside the figure; the vertical version still fits too;
* every expression written in a figure is explained by ``math_notes``;
* the covariate gain tables have no empty cells and list a model only where it read covariates;
* the actual-vs-forecast plots and tables show every series of a multivariate dataset when asked for "all".
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

from ch03_zeroshot.architectures import ARCH, MATH, figure, math_notes  # noqa: E402
from ch03_zeroshot.diagrams import FONT_RC  # noqa: E402
from ch03_zeroshot.report import (forecast_gallery, gain_agreement, gain_tables, plot_horizons,  # noqa: E402
                                  series_list, show_forecast_tables)

plt.rcParams.update(FONT_RC)     # measure text as the saved figures draw it, whatever this machine's defaults are
ON_BOXES = {"callout", "title", "label", "tag", "boxtext"}      # text that may sit on a box (its own or a panel's)


def _problems(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax = fig.axes[0]
    texts = [t for t in ax.texts if t.get_text().strip()]
    ext = {id(t): t.get_window_extent(r) for t in texts}
    boxes = [p.get_window_extent(r) for p in ax.patches if isinstance(p, FancyBboxPatch) and p.get_gid() == "box"]
    fb = fig.bbox
    problems = []
    for t in texts:
        e = ext[id(t)]
        if e.x0 < fb.x0 - 1 or e.x1 > fb.x1 + 1 or e.y0 < fb.y0 - 1 or e.y1 > fb.y1 + 1:
            problems.append(f"outside the figure: {t.get_text()!r}")
        if t.get_gid() == "boxtext":
            own = [b for b in boxes if b.x0 <= (e.x0 + e.x1) / 2 <= b.x1 and b.y0 <= (e.y0 + e.y1) / 2 <= b.y1]
            if not own:
                problems.append(f"no box around: {t.get_text()!r}")
                continue
            b = min(own, key=lambda b: b.width * b.height)
            if e.x0 < b.x0 + 2 or e.x1 > b.x1 - 2 or e.y0 < b.y0 + 1 or e.y1 > b.y1 - 1:
                problems.append(f"overflows its box: {t.get_text()!r}")
        elif t.get_gid() not in ON_BOXES and any(e.overlaps(b) for b in boxes):
            problems.append(f"{t.get_gid()} over a box: {t.get_text()!r}")
    panels = [p.get_window_extent(r) for p in ax.patches if isinstance(p, FancyBboxPatch) and p.get_gid() == "panel"]
    for b in boxes:                       # a layer box sits either fully inside a panel or clear of it
        for pn in panels:
            inside = pn.x0 - 1 <= b.x0 and b.x1 <= pn.x1 + 1 and pn.y0 - 1 <= b.y0 and b.y1 <= pn.y1 + 1
            if b.overlaps(pn) and not inside:
                problems.append(f"a box crosses the edge of a panel at ({b.x0:.0f}, {b.y0:.0f})")
    for i, a in enumerate(texts):
        for c in texts[i + 1:]:
            if ext[id(a)].overlaps(ext[id(c)]) and not {a.get_gid(), c.get_gid()} <= {"boxtext", "callout"}:
                problems.append(f"overlap: {a.get_text()!r} / {c.get_text()!r}")
    plt.close(fig)
    return problems


@pytest.mark.parametrize("key", list(ARCH))
def test_figure_text_fits(key):
    assert not _problems(figure(key, show=True, style="classic"))


def _print_problems(fig, mx=0.05, my=0.03):
    """Stricter checks for figures that go into the book: box text keeps a margin of ``mx`` / ``my`` inches,
    no line or arrow runs through any text, and no numbered circle sits on text."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax = fig.axes[0]
    dpi = fig.dpi
    texts = [t for t in ax.texts if t.get_text().strip()]
    ext = {id(t): t.get_window_extent(r) for t in texts}
    boxes = [p.get_window_extent(r) for p in ax.patches if isinstance(p, FancyBboxPatch) and p.get_gid() == "box"]
    problems = []
    for t in texts:
        if t.get_gid() != "boxtext":
            continue
        e = ext[id(t)]
        own = [b for b in boxes if b.x0 <= (e.x0 + e.x1) / 2 <= b.x1 and b.y0 <= (e.y0 + e.y1) / 2 <= b.y1]
        b = min(own, key=lambda b: b.width * b.height)
        if min(e.x0 - b.x0, b.x1 - e.x1) / dpi < mx or min(e.y0 - b.y0, b.y1 - e.y1) / dpi < my:
            problems.append(f"text too close to its box edge: {t.get_text()!r}")
    paths = [ax.transData.transform(ln.get_xydata()) for ln in ax.lines if len(ln.get_xydata()) > 1]
    paths += [p.get_path().vertices for p in ax.patches if isinstance(p, FancyArrowPatch)]
    for t in texts:
        if t.get_gid() == "legend":
            continue
        e = ext[id(t)]
        for d in paths:
            hit = False
            for (x0, y0), (x1, y1) in zip(d[:-1], d[1:]):
                n = int(max(abs(x1 - x0), abs(y1 - y0)) / 2) + 2
                xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
                if np.any((xs > e.x0 + 1) & (xs < e.x1 - 1) & (ys > e.y0 + 1) & (ys < e.y1 - 1)):
                    hit = True
                    break
            if hit:
                problems.append(f"a line runs through: {t.get_text()!r}")
                break
    circles = [p.get_window_extent(r) for p in ax.patches
               if isinstance(p, Circle) and tuple(p.get_facecolor()[:3]) != (1.0, 1.0, 1.0)]
    for c in circles:
        problems += [f"a numbered circle sits on: {t.get_text()!r}" for t in texts
                     if t.get_gid() != "callout" and c.overlaps(ext[id(t)])]
    plt.close(fig)
    return problems


@pytest.mark.parametrize("key", list(ARCH))
def test_figure_print_quality(key):
    assert not _print_problems(figure(key, show=True, style="classic"))


@pytest.mark.parametrize("key", list(ARCH))
def test_explainer_layout_is_clean(key, tmp_path):
    """The default notebook figure is the step-by-step explainer: no text outside its panel and no overlaps, and
    it is saved as PNG and PDF."""
    from ch03_zeroshot.explainers import SPECS
    assert key in SPECS
    fig = figure(key, path=tmp_path / f"{key}.png", show=False)
    assert not fig._layout_issues, fig._layout_issues
    assert (tmp_path / f"{key}.png").stat().st_size > 0 and (tmp_path / f"{key}.pdf").stat().st_size > 0


@pytest.mark.parametrize("hinting", ["force_autohint", "native", "no_hinting"])
def test_explainer_does_not_depend_on_text_hinting(hinting):
    """Stock matplotlib (Colab, a Mac) hints text differently from some Linux set-ups; the explainer pins its own
    hinting, so the layout and the picture are the same everywhere."""
    import io

    import matplotlib as mpl

    from ch03_zeroshot.explainers import figure as explainer
    from ch03_zeroshot.explainers.explainer import RC

    pngs = []
    for setting in (hinting, "no_hinting"):
        with mpl.rc_context({"text.hinting": setting}):
            fig = explainer("ttm", show=False)
            assert not fig._layout_issues, fig._layout_issues
            buf = io.BytesIO()
            with mpl.rc_context(RC):
                fig.savefig(buf, format="png", dpi=100, facecolor="white")
            pngs.append(buf.getvalue())
        plt.close(fig)
    assert pngs[0] == pngs[1]


@pytest.mark.parametrize("key", ["chronos2", "chronos-bolt", "timemoe"])
def test_vertical_figure_still_fits(key):
    assert not _problems(figure(key, show=True, vertical=True))


@pytest.mark.parametrize("key", list(ARCH))
def test_every_expression_is_explained(key):
    fig = figure(key, show=False, style="classic")
    notes = math_notes(key)
    for k in fig.used_math:
        assert k in MATH and MATH[k][0] in notes, k


def _agg():
    rows = []
    for d, hs in (("D1", (7, 14)), ("D2", (7, 14)), ("D3", (3, 6)), ("D4", (3, 6))):
        for h in hs:
            for m, val in (("reads", 1.0), ("ignores", 1.0), ("arima", 1.0)):
                bump = 0.9 if (m == "reads" and d in ("D2", "D4")) else 1.0
                rows.append({"dataset": d, "series": "s", "horizon": h, "model": m, "MASE": val * bump,
                             "RMSSE": val * bump * 1.02})
            if d in ("D2", "D4"):
                rows.append({"dataset": d, "series": "s", "horizon": h, "model": "arimax", "MASE": 0.8, "RMSSE": 0.85})
    return pd.DataFrame(rows)


def test_gain_tables_split_by_grain_without_empty_cells():
    g = gain_tables(_agg(), names={"arimax": "ARIMAX", "arima": "ARIMA"})
    assert set(g) == {"daily", "monthly"}
    assert list(g["daily"].columns.get_level_values(1).unique()) == [7, 14]
    assert list(g["monthly"].columns.get_level_values(1).unique()) == [3, 6]
    for t in g.values():
        assert not t.isna().any().any()
        assert list(t.columns.get_level_values(0).unique()) == ["MASE", "RMSE"]
        models = set(t.index.get_level_values("model"))
        assert "ignores" not in models and "ARIMA" not in models           # identical forecasts: no gain to report
        assert {"reads", "ARIMAX (vs ARIMA)"} <= models                   # covariate-only model vs its plain twin
    assert np.allclose(g["daily"].loc[("D2 vs D1", "reads"), "MASE"], -10.0)
    assert np.allclose(g["daily"].loc[("D2 vs D1", "ARIMAX (vs ARIMA)"), "MASE"], -20.0)
    s = gain_agreement(g)
    assert (s["metrics disagree at"] == 0).all() and (s["horizons"] == 2).all()


def _forecasts():
    """Two monthly series (stored in the order zinc, copper) with two horizons, and their frame; one model reads
    covariates on D4 (different forecasts from D3), the other does not."""
    idx = pd.date_range("2020-01-01", periods=40, freq="MS")
    frame = pd.DataFrame({"zinc": np.arange(40.0), "copper": np.arange(40.0) * 2}, index=idx)
    rows = []
    for d in ("D3", "D4"):
        for s in ("zinc", "copper"):
            for h in (3, 6):
                for k in range(h):
                    ds = idx[40 - h + k]
                    for m, shift in (("reads", 1.0 if d == "D4" else 0.0), ("plain", 0.0)):
                        rows.append({"dataset": d, "series": s, "horizon": h, "window": 1, "step": k + 1, "ds": ds,
                                     "model": m, "y_true": frame.loc[ds, s], "y_pred": frame.loc[ds, s] + 1 + shift})
    return pd.DataFrame(rows), frame


def test_series_list_keeps_the_stored_order():
    fc, frame = _forecasts()
    assert series_list(fc, "D3", "all") == ["zinc", "copper"]
    assert series_list(fc, "D3", "all", frame[["copper", "zinc"]]) == ["copper", "zinc"]
    assert series_list(fc, "D3", "copper") == ["copper"]
    assert series_list(fc, "D3", ["copper", "zinc"]) == ["copper", "zinc"]


def test_forecast_plots_show_every_series():
    fc, frame = _forecasts()
    fig = plot_horizons(fc, frame, "D3", "all", ["reads", "plain"])
    axes = [a for a in fig.axes if a.get_visible()]
    assert len(axes) == 4                                              # 2 series x 2 horizons
    titles = sorted(a.get_title() for a in axes)
    assert titles == ["copper · H = 3 months", "copper · H = 6 months", "zinc · H = 3 months", "zinc · H = 6 months"]
    one = plot_horizons(fc, frame, "D3", "copper", ["reads", "plain"])
    assert [a.get_title() for a in one.axes] == ["H = 3 months", "H = 6 months"]
    plt.close("all")
    figs = forecast_gallery(fc, {"D3": frame, "D4": frame}, {"D3": "all"}, ["reads", "plain"], show=False)
    assert len(figs["D3"].axes) == 4 and len(figs["D4"].axes) == 4    # a dataset left out of SERIES shows all
    drawn = {t.get_text() for t in figs["D4"].legends[0].get_texts()}
    assert "reads" in drawn and "plain" not in drawn                   # D4 draws only the model that read covariates
    plt.close("all")


def test_forecast_tables_cover_every_series(tmp_path, monkeypatch):
    import IPython.display

    shown = []
    monkeypatch.setattr(IPython.display, "display", lambda *a, **k: shown.append(a[0]))
    fc, _ = _forecasts()
    out = tmp_path / "t.csv"
    show_forecast_tables(fc, {"D3": "all"}, {"D3": 3}, ["reads", "plain"], save=out)
    titles = [x.data for x in shown if hasattr(x, "data") and isinstance(x.data, str)]
    assert len(titles) == 2 and "zinc" in titles[0] and "copper" in titles[1]
    saved = pd.read_csv(out)
    assert set(saved["series"]) == {"zinc", "copper"} and set(saved["horizon"]) == {3, 6}
