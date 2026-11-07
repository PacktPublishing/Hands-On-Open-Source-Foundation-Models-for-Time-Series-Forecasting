"""Small helpers the forecasting notebooks share: load stored forecasts, score them, compare covariate gains,
plot forecasts against the actual values."""
# --- cell: report ---
from pathlib import Path

import numpy as np
import pandas as pd

from ch03_zeroshot.metrics import SCALE_LAG, aggregate, evaluate


def load_forecasts(root, models=None, datasets=None, smoke=False):
    """Every stored forecast (long format) of ``models`` on ``datasets`` (all when None)."""
    base = Path(root) / "results" / ("forecasts_smoke" if smoke else "forecasts")
    parts = []
    for d in sorted(base.iterdir()) if base.exists() else []:
        if datasets is not None and d.name not in datasets:
            continue
        for f in sorted(d.glob("*.csv")):
            if models is not None and f.stem not in models:
                continue
            if f.stat().st_size > 1:                     # a run that produced no forecast leaves an empty file
                parts.append(pd.read_csv(f, parse_dates=["ds"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def score(root, forecasts):
    """Per-forecast metrics (one row per model, series, horizon, window) for every dataset present."""
    return pd.concat([evaluate(g, root, name) for name, g in forecasts.groupby("dataset")], ignore_index=True)


def scores_cached(root, models=None, datasets=None, refresh=False):
    """Per-forecast metrics of the stored full runs, cached in results/tables/scores_per_forecast.csv."""
    path = Path(root) / "results" / "tables" / "scores_per_forecast.csv"
    fc = load_forecasts(root, models, datasets)
    if path.exists() and not refresh:
        cached = pd.read_csv(path)
        have = set(map(tuple, cached[["dataset", "model"]].drop_duplicates().to_numpy()))
        need = set(map(tuple, fc[["dataset", "model"]].drop_duplicates().to_numpy()))
        missing = need - have
        if missing:
            add = fc[[tuple(x) in missing for x in fc[["dataset", "model"]].to_numpy()]]
            cached = pd.concat([cached, score(root, add)], ignore_index=True)
            cached.to_csv(path, index=False)
    else:
        cached = score(root, fc)
        path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(path, index=False)
    keep = np.ones(len(cached), bool)
    if models is not None:
        keep &= cached["model"].isin(models).to_numpy()
    if datasets is not None:
        keep &= cached["dataset"].isin(datasets).to_numpy()
    return cached[keep].reset_index(drop=True)


def horizon_table(agg, metric="MASE", models=None, names=None):
    """Rows (dataset, horizon), columns models: the metric averaged over series and windows."""
    d = agg if models is None else agg[agg["model"].isin(models)]
    t = d.groupby(["dataset", "horizon", "model"])[metric].mean().unstack("model")
    if models is not None:
        t = t[[m for m in models if m in t.columns]]
    return t.rename(columns=names or {})


DASH = "–"   # shown where a model has no forecast (not run on that dataset), instead of NaN


def show_horizon_table(agg, metric="MASE", models=None, names=None, digits=3):
    """horizon_table as the notebooks show it: best value of each row in bold, a dash where a model was not run."""
    t = horizon_table(agg, metric, models, names)
    return t.style.format(f"{{:.{digits}f}}", na_rep=DASH).highlight_min(axis=1, props="font-weight: bold")


def _status(entry):
    """Short status label and an explanation for one dataset of a run log."""
    status, notes = entry.get("status") or "", list(entry.get("notes", [])) + list(entry.get("skipped", []))
    if status.startswith("same input as"):
        twin = status.split()[3]
        return f"{twin} forecasts reused", [f"reads none of these covariates, so its input is exactly {twin}'s"] + notes
    if status.startswith("skipped"):
        return "not run", [status.split(": ", 1)[-1] + " (its forecast would repeat D3's)"] + notes
    return {"ok": "ran", "partial": "ran, some windows skipped"}.get(status, status), notes


def run_table(log, datasets):
    """One row per dataset of a run log: status, forecasting seconds, covariates read, notes; no NaN or cut-off text."""
    rows = []
    for d in datasets:
        if d not in log["datasets"]:
            continue
        e = log["datasets"][d]
        status, notes = _status(e)
        row = {"dataset": d, "status": status, "seconds": e.get("seconds"),
               "covariates read": ", ".join(e.get("known_future", []) + e.get("past_only", [])) or "none"}
        if "hyperparameters" in e:          # learned baselines: the setting chosen on the validation windows
            row["hyperparameters"] = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                                               for k, v in e["hyperparameters"].items()) or "defaults"
        rows.append({**row, "notes": "; ".join(notes)})
    t = pd.DataFrame(rows).set_index("dataset")
    text = [c for c in ("status", "covariates read", "hyperparameters", "notes") if c in t.columns]
    return t.style.format({"seconds": "{:.1f}"}, na_rep=DASH).set_properties(subset=text, **{"text-align": "left"})


def cost_table(runs, datasets):
    """Forecasting seconds by model and dataset; a dash where the model was not run, 0.0 where forecasts were reused."""
    t = runs[runs["dataset"].isin(datasets)].pivot_table(index="model", columns="dataset", values="seconds", dropna=False)
    t = t[[d for d in datasets if d in t.columns]]
    return t.style.format("{:.1f}", na_rep=DASH)


def covariate_gain(agg, metric="MASE", pairs=(("D2", "D1"), ("D4", "D3"))):
    """Relative change of ``metric`` when a model reads the covariates: 100 * (with / without - 1).
    Negative = the covariates helped. Only models that actually read covariates on the richer dataset."""
    rows = []
    for rich, plain in pairs:
        a = agg[agg["dataset"] == rich].groupby(["model", "horizon"])[metric].mean()
        b = agg[agg["dataset"] == plain].groupby(["model", "horizon"])[metric].mean()
        both = pd.concat([a.rename("with"), b.rename("without")], axis=1).dropna()
        both = both[(both["with"] - both["without"]).abs() > 1e-12]       # identical = covariates not read
        both["change_%"] = 100 * (both["with"] / both["without"] - 1)
        both["comparison"] = f"{rich} vs {plain}"
        rows.append(both.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


TWINS = {"arimax": "arima", "ardl": "ar", "varx": "var"}   # baselines that need covariates, and their plain twin


def gain_tables(agg, models=None, names=None, twins=TWINS):
    """Covariate gain by MASE and by RMSE, one table per time grain (no empty cells: the grains have different horizons).

    Rows (comparison, model), columns (metric, horizon), values: % change when the model reads the covariates
    (negative = they helped). A model appears only in the comparisons where it read covariates. The RMSE change is
    measured on the RMSE scaled like MASE (RMSSE), so that every series counts equally; for a single series, such as
    temperature, it is the change in RMSE. A model that runs only with covariates (``twins``: ARIMAX, ARDL, VARX) is
    compared with its plain twin on D1 or D3 (ARIMA, AR, VAR).
    """
    d = agg if models is None else agg[agg["model"].isin(models)]
    names = dict(names or {})
    for rich, plain in twins.items():
        if rich in set(d["model"]) and plain in set(agg["model"]):
            twin = agg[(agg["model"] == plain) & agg["dataset"].isin(["D1", "D3"])].assign(model=rich)
            d = pd.concat([d[~((d["model"] == rich) & d["dataset"].isin(["D1", "D3"]))], twin], ignore_index=True)
            names[rich] = f"{names.get(rich, rich)} (vs {names.get(plain, plain)})"
    out = {}
    for grain, pairs in (("daily", (("D2", "D1"),)), ("monthly", (("D4", "D3"),))):
        parts = [covariate_gain(d, metric, pairs).assign(metric=label)
                 for metric, label in (("MASE", "MASE"), ("RMSSE", "RMSE"))]
        g = pd.concat([x for x in parts if len(x)], ignore_index=True) if any(len(x) for x in parts) else None
        if g is None:
            continue
        g["model"] = g["model"].map(lambda m: names.get(m, m))
        t = g.pivot_table(index=["comparison", "model"], columns=["metric", "horizon"], values="change_%")
        out[grain] = t.reindex(columns=["MASE", "RMSE"], level=0)
    return out


def gain_agreement(tables):
    """Per (comparison, model): mean % change by MASE and by RMSE, and at how many horizons each metric improves."""
    rows = []
    for t in tables.values():
        for idx, r in t.iterrows():
            a, b = r["MASE"], r["RMSE"]
            ok = a.notna() & b.notna()
            rows.append({"comparison": idx[0], "model": idx[1],
                         "MASE: mean change %": a[ok].mean(), "RMSE: mean change %": b[ok].mean(),
                         "MASE better at": int((a[ok] < 0).sum()), "RMSE better at": int((b[ok] < 0).sum()),
                         "metrics disagree at": int(((a[ok] < 0) != (b[ok] < 0)).sum()), "horizons": int(ok.sum())})
    return pd.DataFrame(rows).set_index(["comparison", "model"]) if rows else pd.DataFrame()


def show_gains(tables, limit=25):
    """Display the gain tables colour-coded (green = the covariates helped, red = they hurt) and the summary."""
    from IPython.display import Markdown, display

    if not tables:
        print("No model here reads covariates, so D2 and D4 repeat the D1 and D3 forecasts.")
        return
    for grain, t in tables.items():
        title = ("Daily temperature, D2 against D1 (horizons in days)" if grain == "daily"
                 else "Monthly metals, D4 against D3 (horizons in months)")
        display(Markdown(f"**{title}**: % change in MASE and in RMSE when the model reads the covariates"))
        display(t.style.format("{:+.1f}", na_rep="n/a")
                .background_gradient(cmap="RdYlGn_r", vmin=-limit, vmax=limit, axis=None))
    s = gain_agreement(tables)
    display(Markdown("**MASE against RMSE**: mean change over the horizons, and at how many horizons each metric "
                     "improves"))
    display(s.style.format({c: "{:+.1f}" for c in s.columns if "%" in c})
            .background_gradient(cmap="RdYlGn_r", vmin=-limit, vmax=limit,
                                 subset=[c for c in s.columns if "%" in c]))


UNIT = {"D1": "days", "D2": "days", "B1": "trading days", "B2": "trading days"}
PLAIN = {"D2": "D1", "D4": "D3"}


def series_list(forecasts, dataset, spec="all", frame=None):
    """The series that ``spec`` names on ``dataset``: one name, a list of names, or "all" (every series with a stored
    forecast, in the column order of ``frame`` when it is given, otherwise in the stored order, which is the same)."""
    if isinstance(spec, str) and spec != "all":
        return [spec]
    if spec is not None and not isinstance(spec, str):
        return list(spec)
    present = list(pd.unique(forecasts.loc[forecasts["dataset"] == dataset, "series"]))
    order = [c for c in (frame.columns if frame is not None else []) if c in present]
    return order + [c for c in present if c not in order]


def plot_horizons(forecasts, frame, dataset, series="all", models=None, names=None, window=1, colors=None, dashed=()):
    """One panel per horizon and one row of panels per series (``series``: a name, a list or "all"), all from the
    same test window: recent history, actual values, every model's forecast (models in ``dashed`` drawn with dashed
    lines)."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    names, colors = names or {}, colors or {}
    rows = series_list(forecasts, dataset, series, frame)
    f = forecasts[(forecasts["dataset"] == dataset) & (forecasts["series"].isin(rows))
                  & (forecasts["window"] == window)]
    if models is not None:
        f = f[f["model"].isin(models)]
    rows = [s for s in rows if s in set(f["series"])]
    horizons = sorted(f["horizon"].unique())
    if not horizons:
        print(f"{dataset} {', '.join(series_list(forecasts, dataset, series, frame))}: "
              f"no stored forecast for window {window}")
        return None
    order = [m for m in (models or sorted(f["model"].unique())) if m in set(f["model"])]
    legend_rows = -(-(len(order) + 2) // 9)
    legend_in = 0.22 * legend_rows + 0.15
    height = 2.6 * len(rows) + 0.3 + legend_in
    fig, axes = plt.subplots(len(rows), len(horizons), figsize=(3.3 * len(horizons), height), squeeze=False)
    unit = UNIT.get(dataset, "months")
    for r, s in enumerate(rows):
        for ax, h in zip(axes[r], horizons):
            g = f[(f["series"] == s) & (f["horizon"] == h)]
            if g.empty:
                ax.set_axis_off()
                continue
            start = g["ds"].min()
            past = frame[s].loc[:start].iloc[-max(2 * h, 30 if dataset in UNIT else 24) - 1:-1]
            ax.plot(past.index, past.to_numpy(), color="#9a9a9a", lw=0.9, label="history")
            test = g.drop_duplicates("ds").sort_values("ds")
            ax.plot([past.index[-1], *test["ds"]], [past.iloc[-1], *test["y_true"]], color="#111111", lw=1.6,
                    label="actual")
            for m in order:
                gm = g[g["model"] == m].sort_values("ds")
                ax.plot(gm["ds"], gm["y_pred"], lw=1.1, color=colors.get(m), ls="--" if m in dashed else "-",
                        label=names.get(m, m))
            ax.axvline(start, color="#bbbbbb", lw=0.7, ls="--")
            ax.set_title(f"{s} · H = {h} {unit}" if len(rows) > 1 else f"H = {h} {unit}", fontsize=9)
            loc = mdates.AutoDateLocator(maxticks=4)
            loc.intervald[mdates.MONTHLY] = [1, 2, 3, 4, 6, 12, 24]   # years of monthly data: no crowded ticks
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
            ax.xaxis.get_offset_text().set_fontsize(7)
            ax.tick_params(labelsize=7)
    seen = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, handle)
    fig.legend(list(seen.values()), list(seen), loc="lower center", ncol=min(len(seen), 9), fontsize=7.5,
               frameon=False)
    what = rows[0] if len(rows) == 1 else (f"all {len(rows)} series" if series == "all" else ", ".join(rows))
    fig.suptitle(f"{dataset} · {what} · test window {window}" + (" (the most recent)" if window == 1 else ""),
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, legend_in / height, 1, 1))
    return fig


def model_colors(models):
    """One fixed colour per model, so a model keeps its colour in every panel and figure."""
    import matplotlib.pyplot as plt

    if len(models) <= 9:                     # tab10 without its grey, which is kept for the history
        palette = [c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 7]
    else:                                    # tab20 without greys, the strong shades first
        tab20 = plt.get_cmap("tab20").colors
        palette = [tab20[i] for i in (0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 3, 5, 7, 9, 11, 13, 17, 19)]
    return {m: palette[i % len(palette)] for i, m in enumerate(models)}


def forecast_gallery(forecasts, frames, series, models, names=None, window=1, datasets=None, show=True):
    """``plot_horizons`` for every dataset, with the series in ``series[dataset]`` (a name, a list or "all"; a
    dataset left out of ``series`` shows all its series). On D2 and D4 only the models that read covariates are
    drawn: the others' forecasts are the same as on D1 or D3."""
    import matplotlib.pyplot as plt

    colors = model_colors(models)
    figs = {}
    for d in datasets or list(frames):
        spec = series.get(d, "all")
        ms = list(models)
        if d in PLAIN:
            names_d = series_list(forecasts, d, spec, frames[d])
            ms = [m for m in ms if _reads_covariates(forecasts, d, PLAIN[d], m, names_d, window)]
            if not ms:
                print(f"{d}: no model here reads its covariates, so its forecasts are the {PLAIN[d]} ones above.")
                continue
        figs[d] = plot_horizons(forecasts, frames[d], d, spec, ms, names, window, colors)
        if show and figs[d] is not None:
            plt.show()
    return figs


def _reads_covariates(forecasts, rich, plain, model, series, window):
    """True when ``model`` has forecasts on ``rich`` that differ from its forecasts on ``plain`` (``series``: one name
    or a list of names)."""
    series = [series] if isinstance(series, str) else list(series)
    sel = (forecasts["model"] == model) & (forecasts["series"].isin(series)) & (forecasts["window"] == window)
    key = ["series", "horizon", "step"]
    a = forecasts[sel & (forecasts["dataset"] == rich)].sort_values(key)["y_pred"].to_numpy()
    b = forecasts[sel & (forecasts["dataset"] == plain)].sort_values(key)["y_pred"].to_numpy()
    return len(a) > 0 and (len(a) != len(b) or not np.allclose(a, b, rtol=1e-6, atol=1e-9))


def forecast_table(forecasts, dataset, series, horizon, models, window=1):
    """Actual values and each model's point forecast, step by step, for one series, horizon and test window.

    Returns (steps, errors): ``steps`` has one row per forecast date (actual first, then the models); ``errors`` has
    the MAE, the RMSE and the mean error ME = mean(actual - forecast) of each model over the window (ME > 0: the
    forecast was too low on average)."""
    f = forecasts[(forecasts["dataset"] == dataset) & (forecasts["series"] == series)
                  & (forecasts["horizon"] == horizon) & (forecasts["window"] == window)]
    if f.empty:
        return None, None
    wide = f.pivot_table(index="ds", columns="model", values="y_pred")
    wide = wide[[m for m in models if m in wide.columns]]
    steps = pd.concat([f.drop_duplicates("ds").set_index("ds")["y_true"].rename("actual"), wide], axis=1).sort_index()
    e = -(steps[wide.columns].sub(steps["actual"], axis=0))           # actual - forecast
    errors = pd.DataFrame({"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()), "ME (actual - forecast)": e.mean()}).T
    return steps, errors


def show_forecast_tables(forecasts, series, horizons, models, names=None, window=1, datasets=None,
                         keep=("naive", "snaive"), save=None):
    """For every dataset and every series in ``series[dataset]`` (a name, a list or "all"): the forecast table at
    ``horizons[dataset]``, the closest forecast of each date in bold and the error rows underneath (lowest MAE and
    RMSE, smallest |ME| in bold). On D2 and D4 only the models that read covariates are listed, with the reference
    rules in ``keep``. ``save``: a CSV path for every horizon of ``window`` (all datasets and series, long format)."""
    names = names or {}
    datasets = datasets or list(series)
    rows = []
    for d in datasets:
        targets = series_list(forecasts, d, series.get(d, "all"))
        ms = list(models)
        if d in PLAIN:
            ms = [m for m in ms if m in keep or _reads_covariates(forecasts, d, PLAIN[d], m, targets, window)]
            if not [m for m in ms if m not in keep]:
                continue
        for target in targets:
            steps, errors = forecast_table(forecasts, d, target, horizons[d], ms, window)
            if steps is None:
                print(f"{d} {target}: no stored forecast for horizon {horizons[d]}, window {window}")
                continue
            _show_steps(steps, errors, names, f"**{d} · {target} · H = {horizons[d]} {UNIT.get(d, 'months')} · "
                        f"test window {window}** ({steps.index.min():%Y-%m-%d} to {steps.index.max():%Y-%m-%d})")
            if save is not None:
                for h in sorted(forecasts.loc[forecasts["dataset"] == d, "horizon"].unique()):
                    s, _ = forecast_table(forecasts, d, target, h, ms, window)
                    if s is not None:
                        rows.append(s.rename(columns=lambda m: names.get(m, m)).reset_index()
                                    .assign(dataset=d, series=target, horizon=h, window=window))
    if save is not None and rows:
        out = pd.concat(rows, ignore_index=True)
        lead = ["dataset", "series", "horizon", "window", "ds", "actual"]
        out = out[lead + [c for c in out.columns if c not in lead]]
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save, index=False)
        print(f"Every horizon of window {window} saved to {save}")


def _show_steps(steps, errors, names, title):
    """Display one forecast table with its heading: the closest forecast of each date and the best error values in
    bold, actual values in blue, error rows in grey."""
    from IPython.display import Markdown, display

    display(Markdown(title))
    digits = 0 if steps["actual"].abs().median() > 1000 else 2
    shown = pd.concat([steps.rename(index=lambda x: f"{x:%Y-%m-%d}"), errors])
    shown = shown.rename(columns=lambda m: names.get(m, m))
    n = len(steps)
    cols = shown.columns[1:]

    def mark(df):
        out = pd.DataFrame("", index=df.index, columns=df.columns)
        gap = df.iloc[:n][cols].sub(df.iloc[:n]["actual"], axis=0).abs()
        for i, c in gap.idxmin(axis=1).items():
            out.loc[i, c] = "font-weight: bold"
        for r in ("MAE", "RMSE"):
            out.loc[r, df.loc[r, cols].astype(float).idxmin()] = "font-weight: bold"
        me = "ME (actual - forecast)"
        out.loc[me, df.loc[me, cols].astype(float).abs().idxmin()] = "font-weight: bold"
        out.iloc[:n, 0] = "background-color: #eaf2fb"
        out.iloc[n:] = out.iloc[n:].replace("", "background-color: #f2f2f2") \
            .map(lambda s: s if "background" in s else s + "; background-color: #f2f2f2")
        return out

    display(shown.style.format(f"{{:,.{digits}f}}", na_rep="").apply(mark, axis=None))


def plot_window(forecasts, frame, dataset, series, horizon, window=1, models=None, names=None, ax=None,
                history=None, colors=None):
    """Actuals around one test window and the point forecasts of ``models``."""
    import matplotlib.pyplot as plt

    names, colors = names or {}, colors or {}
    f = forecasts[(forecasts["dataset"] == dataset) & (forecasts["series"] == series)
                  & (forecasts["horizon"] == horizon) & (forecasts["window"] == window)]
    if models is not None:
        f = f[f["model"].isin(models)]
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.4))
    start = f["ds"].min()
    history = history or (3 * horizon if horizon > 12 else 4 * horizon)
    past = frame[series].loc[:start].iloc[-history - 1:-1]
    ax.plot(past.index, past.to_numpy(), color="#555555", lw=1.0, label="history")
    test = f.drop_duplicates("ds").sort_values("ds")
    ax.plot(test["ds"], test["y_true"], color="#111111", lw=1.6, label="actual")
    for m, g in f.groupby("model", sort=False):
        g = g.sort_values("ds")
        ax.plot(g["ds"], g["y_pred"], lw=1.2, label=names.get(m, m), color=colors.get(m))
    ax.axvline(start, color="#999999", lw=0.8, ls="--")
    ax.set_title(f"{dataset} {series}, H = {horizon}, window {window}", fontsize=10)
    ax.legend(fontsize=7, ncol=4, frameon=False)
    return ax
