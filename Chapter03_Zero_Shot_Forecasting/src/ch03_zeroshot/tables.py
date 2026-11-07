"""Results tables in the layout of Sengupta, Chakraborty and Singh (2025, IJF, Tables 5-6) and the
Uncertainty-Augmented Bayesian VAR manuscript (Tables 5-6): variable, horizon and metric down the
rows, one column per model, the best value in bold and the second best in italic underlined.
"""
# --- cell: tables ---
import numpy as np
import pandas as pd

from ch03_zeroshot.metrics import CATALOGUE, TABLE_METRICS

# lower is better for every table metric; for ME and FVA the rule is different
BEST = {"ME": "abs_min", "FVA": "max", "TS": "abs_min"}


def display_names():
    """Model key -> name printed in tables and figures."""
    from ch03_zeroshot.capabilities import ZERO_SHOT_MODELS

    names = {r[0]: r[1] for r in ZERO_SHOT_MODELS}
    try:
        from ch03_zeroshot.baselines import BASELINES
        names.update({k: v["name"] for k, v in BASELINES.items()})
    except ImportError:
        pass
    return names


def paper_table(agg, dataset, models=None, metrics=TABLE_METRICS, decimals=3):
    """Wide table for one dataset: rows (Variable, Horizon, Metric), columns models (display names)."""
    d = agg[agg["dataset"] == dataset]
    if models is not None:
        d = d[d["model"].isin(models)]
    names = display_names()
    long = d.melt(id_vars=["series", "horizon", "model"], value_vars=list(metrics), var_name="metric")
    t = long.pivot_table(index=["series", "horizon", "metric"], columns="model", values="value", aggfunc="first")
    order = models or list(t.columns)
    t = t[[m for m in order if m in t.columns]]
    series = list(dict.fromkeys(d["series"]))
    idx = pd.MultiIndex.from_product([series, sorted(d["horizon"].unique()), list(metrics)],
                                     names=["Variable", "Horizon", "Metric"])
    t = t.reindex(idx).dropna(how="all")
    t.columns = [names.get(c, c) for c in t.columns]
    t = t.rename(index={m: CATALOGUE[m][0] if m in CATALOGUE else m for m in metrics}, level="Metric")
    return t.round(decimals)


def rank_marks(t):
    """'best' / 'second' / '' for every cell, row by row."""
    marks = pd.DataFrame("", index=t.index, columns=t.columns)
    labels = {v[0]: k for k, v in CATALOGUE.items()}
    for i, row in t.iterrows():
        key = labels.get(i[-1], i[-1])
        rule = BEST.get(key, "min")
        v = row.astype(float)
        v = v.abs() if rule == "abs_min" else (-v if rule == "max" else v)
        order = v.dropna().sort_values()
        uniq = order.unique()
        if len(uniq) >= 1:
            marks.loc[i, order.index[order == uniq[0]]] = "best"
        if len(uniq) >= 2:
            marks.loc[i, order.index[order == uniq[1]]] = "second"
    return marks


def style_table(t, decimals=3):
    """pandas Styler: best bold, second best italic and underlined (as in the papers)."""
    marks = rank_marks(t)
    css = marks.replace({"best": "font-weight: bold", "second": "font-style: italic; text-decoration: underline"})
    return (t.style.format(f"{{:.{decimals}f}}", na_rep="-").apply(lambda _: css, axis=None)
            .set_table_styles([{"selector": "th", "props": "font-size: 9pt; text-align: center"},
                               {"selector": "td", "props": "font-size: 9pt; text-align: right"}]))


def to_latex(t, caption, label, decimals=3):
    """booktabs LaTeX with \\textbf (best) and \\underline{\\textit{}} (second best)."""
    marks = rank_marks(t)
    cols = list(t.columns)
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\scriptsize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lll" + "r" * len(cols) + "}", r"\toprule",
             "Variable & Horizon & Metric & " + " & ".join(c.replace("_", r"\_") for c in cols) + r" \\",
             r"\midrule"]
    prev = (None, None)
    for i, row in t.iterrows():
        var, h, metric = i
        cells = []
        for c in cols:
            v = row[c]
            s = "-" if pd.isna(v) else f"{v:.{decimals}f}"
            if marks.loc[i, c] == "best":
                s = f"\\textbf{{{s}}}"
            elif marks.loc[i, c] == "second":
                s = f"\\underline{{\\textit{{{s}}}}}"
            cells.append(s)
        if prev[0] is not None and var != prev[0]:
            lines.append(r"\midrule")
        elif prev[1] is not None and h != prev[1]:
            lines.append(r"\cmidrule(lr){2-" + str(3 + len(cols)) + "}")
        lines.append(" & ".join([str(var) if var != prev[0] else "", str(h) if (var, h) != prev else "",
                                 metric.replace("%", r"\%")] + cells) + r" \\")
        prev = (var, h)
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def to_excel(t, path):
    """Excel copy with the same bold / italic-underline marks."""
    from openpyxl.styles import Font

    marks = rank_marks(t)
    flat = t.reset_index()
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        flat.to_excel(xw, index=False, sheet_name="table")
        ws = xw.sheets["table"]
        k = t.index.nlevels
        for r, i in enumerate(t.index, start=2):
            for c, col in enumerate(t.columns, start=k + 1):
                m = marks.loc[i, col]
                if m == "best":
                    ws.cell(r, c).font = Font(bold=True)
                elif m == "second":
                    ws.cell(r, c).font = Font(italic=True, underline="single")


def save_table(t, out_dir, stem, caption, label):
    """Write CSV, XLSX and LaTeX copies of one table; returns the paths."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.csv", out_dir / f"{stem}.xlsx", out_dir / f"{stem}.tex"]
    t.to_csv(paths[0])
    to_excel(t, paths[1])
    paths[2].write_text(to_latex(t, caption, label))
    return paths


def average_ranks(agg, metric="MASE", by=("dataset", "series", "horizon"), percent=False):
    """Mean rank of each model across (dataset, series, horizon) cells (1 = best).

    ``percent=True`` rescales the ranks of each cell to 0 (best) ... 100 (worst), so that models scored in cells with
    different numbers of competitors (a VAR runs only where there are several series) stay comparable.
    """
    t = agg.pivot_table(index=list(by), columns="model", values=metric)
    r = t.rank(axis=1)
    if percent:
        n = t.notna().sum(axis=1)
        r = r.sub(1, axis=0).div((n - 1).where(n > 1), axis=0) * 100
    return r.mean().sort_values()
