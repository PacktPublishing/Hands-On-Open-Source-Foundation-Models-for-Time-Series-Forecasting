"""Point-forecast metrics (Notebook 11): definitions follow Chapter 3 of Mastering Modern Time Series
Forecasting (MMTSF), and the table metrics follow Sengupta, Chakraborty and Singh (2025, IJF).

Every metric is computed on one forecast: one series, one horizon H, one rolling window. The
in-sample history (everything up to the forecast origin) gives the MASE/RMSSE scale and the naive
benchmark used by the relative metrics (MDRAE, GMRAE, FVA).
"""
# --- cell: metrics ---
import numpy as np
import pandas as pd

# Naive benchmark and MASE/RMSSE scaling lag per dataset. Daily temperature: seasonal naive (same
# day last year), as in Notebook 03. Metal prices, oil and yields: last value (random walk), because
# Notebook 03 found no usable seasonal cycle in them (seasonal naive = naive there).
SCALE_LAG = {"D1": 365, "D2": 365, "D3": 1, "D4": 1, "B1": 1, "B2": 1}

# name: (label, formula, what it measures, strength, limitation, where it is defined)
CATALOGUE = {
    "MAE": ("MAE", r"\frac{1}{H}\sum_{t=1}^{H}|y_t-\hat y_t|",
            "average size of the error, in the units of the series",
            "easy to read; robust to a few large errors; minimised by the median",
            "scale-dependent: cannot be averaged across series of different units", "MMTSF 3.1"),
    "RMSE": ("RMSE", r"\sqrt{\mathrm{MSE}},\qquad \mathrm{MSE} = \frac{1}{H}\sum_{t=1}^{H}(y_t-\hat y_t)^2",
             "typical error with large misses weighted more", "penalises big misses; same units as the data",
             "dominated by a few large errors; scale-dependent", "MMTSF 3.1; Sengupta et al. (2025) Sec. 5.2"),
    "ME": ("ME (bias)", r"\frac{1}{H}\sum_{t=1}^{H}(y_t-\hat y_t)",
           "systematic over- (ME < 0) or under-forecasting (ME > 0)", "shows direction, which no absolute metric does",
           "positive and negative errors cancel: ME = 0 does not mean accurate", "MMTSF 3.1.1"),
    "TS": ("Tracking signal", r"\frac{\mathrm{CFE}}{\mathrm{MAD}},\quad\mathrm{CFE}=\sum_{t=1}^{H}(y_t-\hat y_t),\quad"
                              r"\mathrm{MAD}=\frac{1}{H}\sum_{t=1}^{H}|y_t-\hat y_t|",
           "persistent bias relative to the error size", "a value outside about [-4, 4] flags a biased model",
           "built for monitoring a running forecast; it grows with the horizon (TS = H · ME/MAE), so this chapter "
           "compares the bias ratio ME/MAE (between -1 and 1) instead", "MMTSF 3.1.1.2"),
    "MAPE": ("MAPE (%)", r"\frac{100}{H}\sum_{t=1}^{H}\left|\frac{y_t-\hat y_t}{y_t}\right|",
             "average error as a percentage of the actual value", "familiar to non-specialists",
             "explodes near zero, undefined at zero, meaningless for interval scales such as degrees Celsius; "
             "penalises over-forecasts more", "MMTSF 3.1.2"),
    "WAPE": ("WAPE (%)", r"100\,\frac{\sum_t|y_t-\hat y_t|}{\sum_t|y_t|}",
             "total absolute error relative to the total level", "stable, volume-weighted, no blow-up at single zeros",
             "still needs a meaningful zero; large levels dominate", "MMTSF 3.1.2.1"),
    "sMAPE": ("sMAPE (%)", r"\frac{100}{H}\sum_{t=1}^{H}\frac{|y_t-\hat y_t|}{(|y_t|+|\hat y_t|)/2}",
              "percentage error against the average of actual and forecast", "bounded at 200%; M4 standard",
              "not symmetric after all: punishes under-forecasts more; unstable near zero", "MMTSF 3.1.2.2"),
    "MAAPE": ("MAAPE (rad)", r"\frac{1}{H}\sum_{t=1}^{H}\arctan\left|\frac{y_t-\hat y_t}{y_t}\right|",
              "percentage error squashed into [0, π/2]", "bounded, so a near-zero actual cannot dominate",
              "units (radians) are not intuitive; rarely reported", "MMTSF 3.1.2.3"),
    "MDAPE": ("MDAPE (%)", r"100\cdot\mathrm{median}_t\left|\frac{y_t-\hat y_t}{y_t}\right|",
              "typical percentage error", "the median ignores a few explosive terms",
              "still undefined at zero and misleading near zero", "Sengupta et al. (2025) Sec. 5.2"),
    "MASE": ("MASE", r"\frac{\frac1H\sum_t|y_t-\hat y_t|}{\frac{1}{T-\ell}\sum_{t=\ell+1}^{T}|y_t-y_{t-\ell}|}",
             "MAE relative to the in-sample error of the naive rule (lag ℓ)", "scale-free, defined for any series; "
             "< 1 means smaller errors than the naive rule made in sample",
             "depends on the naive benchmark: a weak naive makes every model look good",
             "MMTSF 3.1.3; Hyndman and Koehler (2006)"),
    "RMSSE": ("RMSSE", r"\sqrt{\mathrm{MSE}/\mathrm{MSE}_{\text{naive}}},\qquad "
                      r"\mathrm{MSE}_{\text{naive}} = \frac{1}{T-\ell}\sum_{t=\ell+1}^{T}(y_t-y_{t-\ell})^2",
              "RMSE relative to the in-sample naive error", "scale-free and squared-error based (M5 metric)",
              "as MASE, sensitive to the choice of naive", "MMTSF 3.1.3"),
    "TU1": ("Theil's U1", r"\frac{\sqrt{\frac1H\sum_t(y_t-\hat y_t)^2}}{\sqrt{\frac1H\sum_t y_t^2}+\sqrt{\frac1H\sum_t\hat y_t^2}}",
            "RMSE normalised by the size of actuals and forecasts", "bounded in [0, 1]; 0 is a perfect forecast",
            "depends on the level of the series: series far from zero always score close to 0",
            "Theil (1966); Sengupta et al. (2025) Sec. 5.2"),
    "MDRAE": ("MDRAE", r"\mathrm{median}_t\frac{|y_t-\hat y_t|}{|y_t-\hat y^{R}_t|}",
              "typical error relative to the naive forecast over the same window", "scale-free; < 1 beats naive",
              "a step where naive is exactly right gives an infinite ratio (such steps are skipped)",
              "Sengupta et al. (2025) Sec. 5.2"),
    "FVA": ("FVA (%)", r"100\,(1-\mathrm{MAE}/\mathrm{MAE}_{\text{naive}})",
            "how much a model improves on the naive forecast", "direct business reading: +12% better than naive",
            "only as meaningful as the benchmark", "MMTSF 3.2; Gilliland (2013)"),
}

TABLE_METRICS = ["RMSE", "MAE", "MASE", "sMAPE", "TU1", "MDRAE", "MDAPE"]   # Sengupta et al. (2025) + MAE


def naive_forecast(history, horizon, lag):
    """Naive (lag 1) or seasonal-naive (lag = season) forecast from the end of ``history``."""
    history = np.asarray(history, float)
    steps = np.arange(1, horizon + 1)
    return history[len(history) - lag + (steps - 1) % lag]


def _scale(history, lag, power=1):
    d = np.abs(history[lag:] - history[:-lag]) ** power
    d = d[np.isfinite(d)]
    return float(d.mean()) if len(d) else np.nan


def point_metrics(y, yhat, history, lag):
    """All metrics of one forecast. y, yhat: (H,); history: values up to the origin."""
    y, yhat, history = (np.asarray(a, float) for a in (y, yhat, history))
    e = y - yhat
    ae = np.abs(e)
    naive = naive_forecast(history, len(y), lag)
    ae_naive = np.abs(y - naive)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = ae / np.abs(y)
        ape = np.where(np.isfinite(ape), ape, np.nan)
        rae = ae / ae_naive
        rae = np.where(np.isfinite(rae) & (ae_naive > 0), rae, np.nan)
        mad = ae.mean()
        out = {
            "MAE": mad, "MSE": np.mean(e ** 2), "RMSE": np.sqrt(np.mean(e ** 2)), "ME": e.mean(),
            "CFE": e.sum(), "TS": e.sum() / mad if mad > 0 else np.nan, "BIAS": e.mean() / mad if mad > 0 else np.nan,
            "MAPE": 100 * np.nanmean(ape), "MDAPE": 100 * np.nanmedian(ape),
            "WAPE": 100 * ae.sum() / np.abs(y).sum(),
            "sMAPE": 100 * np.mean(np.where(np.abs(y) + np.abs(yhat) > 0, ae / ((np.abs(y) + np.abs(yhat)) / 2), 0.0)),
            "MAAPE": np.nanmean(np.arctan(ape)),
            "MASE": ae.mean() / _scale(history, lag),
            "RMSSE": np.sqrt(np.mean(e ** 2) / _scale(history, lag, 2)),
            "TU1": np.sqrt(np.mean(e ** 2)) / (np.sqrt(np.mean(y ** 2)) + np.sqrt(np.mean(yhat ** 2))),
            "MDRAE": np.nanmedian(rae) if np.isfinite(rae).any() else np.nan,
            "MAE_naive": ae_naive.mean(),
            "log_rae_sum": np.nansum(np.log(rae[np.isfinite(rae) & (rae > 0)])),
            "log_rae_n": int((np.isfinite(rae) & (rae > 0)).sum()),
        }
    out["FVA"] = 100 * (1 - out["MAE"] / out["MAE_naive"]) if out["MAE_naive"] > 0 else np.nan
    return out


def histories(root, dataset):
    """{(series, horizon, window): in-sample values up to the origin} for one dataset."""
    from ch03_zeroshot.data import load_frame, load_manifest

    frame = load_frame(root, dataset)
    out = {}
    for _, r in load_manifest(root, dataset).iterrows():
        past = frame.loc[:r["context_end"]]
        for s in frame.columns:
            out[(s, int(r["horizon"]), int(r["window"]))] = past[s].to_numpy(float)
    return out


def evaluate(forecasts, root, dataset, lag=None):
    """One row of metrics per (model, series, horizon, window) from long-format forecasts."""
    lag = lag or SCALE_LAG[dataset]
    hist = histories(root, dataset)
    rows = []
    for (m, s, h, w), g in forecasts.groupby(["model", "series", "horizon", "window"], sort=False):
        g = g.sort_values("step")
        if g["y_pred"].isna().any():
            continue
        r = point_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), hist[(s, int(h), int(w))], lag)
        rows.append({"dataset": dataset, "model": m, "series": s, "horizon": int(h), "window": int(w), **r})
    return pd.DataFrame(rows)


def aggregate(scores, by=("dataset", "series", "horizon", "model")):
    """Average over rolling windows (MMTSF 3.7.1, uniform averaging); GMRAE pooled geometrically."""
    by = list(by)
    cols = [c for c in scores.columns if c not in by + ["window", "log_rae_sum", "log_rae_n"]
            and pd.api.types.is_numeric_dtype(scores[c])]
    agg = scores.groupby(by)[cols].mean()
    g = scores.groupby(by)[["log_rae_sum", "log_rae_n"]].sum()
    agg["GMRAE"] = np.exp(g["log_rae_sum"] / g["log_rae_n"].where(g["log_rae_n"] > 0))
    return agg.reset_index()


def dtw_distance(a, b):
    """Dynamic time warping distance (absolute-difference cost), a shape diagnostic (MMTSF 3.4)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    cost = np.abs(a[:, None] - b[None, :])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]


def catalogue_table():
    """The metric reference table shown in Notebook 11."""
    return pd.DataFrame([{"metric": v[0], "formula": f"${v[1]}$", "measures": v[2], "strength": v[3],
                          "limitation": v[4], "defined in": v[5]} for v in CATALOGUE.values()])
