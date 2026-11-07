"""Turn a dataset and a split into the inputs a zero-shot model reads (Notebooks 05 to 12)."""
# --- cell: data ---
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ch03_dataprep.config import DATASETS, HORIZONS
from ch03_dataprep.splits import split_frame
from ch03_zeroshot.models.base import SeriesInput

FREQ = {"daily": "D", "monthly": "MS", "trading day": "B"}   # pandas frequency of each grain
SEASON = {"D1": 365, "D2": 365, "D3": 12, "D4": 12, "B1": 5, "B2": 5}   # seasonal-naive period
# Steps in one year. FlowState sets its time scale from it: scale_factor = 24 / steps, the model
# card's rule (0.0656 for daily data without a weekly cycle, 2 for monthly; 252 trading days a year).
YEAR_STEPS = {"D1": 365, "D2": 365, "D3": 12, "D4": 12, "B1": 252, "B2": 252}
DAILY_CONTEXT = 2048     # most recent rows a model reads at a daily origin (monthly: all rows)
MAIN = ["D1", "D2", "D3", "D4"]
BONUS = ["B1", "B2"]
# Known-future covariates are only the ones computed from the date (D2's Fourier terms and day length). Every
# other covariate is past-only: a model sees it up to the forecast origin and never over the test window.
CALENDAR_COVARIATES = ("fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2", "fourier_sin_3",
                       "fourier_cos_3", "daylength_h")


def load_frame(root, name):
    """The frozen dataset (wide CSV in input_timeseries/), indexed by date."""
    return pd.read_csv(Path(root) / "input_timeseries" / f"{name}.csv", index_col=0, parse_dates=True)


def load_manifest(root, name):
    """Split rows of one dataset (written by Notebook 04)."""
    m = pd.read_csv(Path(root) / "data" / "splits" / "split_manifest.csv",
                    parse_dates=["origin_date", "context_start", "context_end", "test_start", "test_end"])
    return m[m["dataset"] == name].reset_index(drop=True)


def validation_rows(frame, manifest, horizon, windows=5):
    """Split rows for ``windows`` back-to-back windows of ``horizon`` steps that end at the earliest test origin of
    the dataset, so they lie inside the training data. A setting is compared on these windows, never on the test
    windows (choosing a setting by its test score would leak the test data into the forecast)."""
    first = manifest["origin_date"].min()
    end = int(np.searchsorted(frame.index, first, side="right"))         # rows up to and including the origin
    rows = []
    for w in range(1, windows + 1):
        t = end - w * horizon
        rows.append({"dataset": manifest["dataset"].iloc[0], "grain": manifest["grain"].iloc[0], "horizon": horizon,
                     "window": w, "primary": False, "origin_date": frame.index[t - 1], "context_start": frame.index[0],
                     "context_end": frame.index[t - 1], "test_start": frame.index[t],
                     "test_end": frame.index[t + horizon - 1], "n_context": t, "n_test": horizon})
    return pd.DataFrame(rows)


def covariate_lists(root, name):
    """(known_future, past_only) covariate columns actually built in Notebook 00.

    Only calendar covariates may be known-future: anything else over the test window would be test data."""
    snap = json.loads((Path(root) / "input_timeseries" / "snapshot_manifest.json").read_text())
    cov = snap["covariates"].get(name, {"known_future": [], "past_only": []})
    leak = [c for c in cov["known_future"] if c not in CALENDAR_COVARIATES]
    if leak:
        raise ValueError(f"{name}: {leak} are not calendar covariates, so their future values are unknown")
    return list(cov["known_future"]), list(cov["past_only"])


def input_plan(root, name, caps):
    """Which columns a model with capabilities ``caps`` receives for dataset ``name``.

    caps: dict with keys multivariate, past_covariates, future_covariates (booleans).
    A covariate the model cannot read is dropped; nothing is ever imputed or invented.
    """
    spec = DATASETS[name]
    known, past = covariate_lists(root, name)
    use_known = known if caps["future_covariates"] else []
    use_past = past if caps["past_covariates"] else []
    targets = list(spec["targets"])
    joint = len(targets) > 1 and caps["multivariate"]
    return {"dataset": name, "grain": spec["grain"], "freq": FREQ[spec["grain"]], "targets": targets,
            "joint": joint, "known_future": use_known, "past_only": use_past,
            "dropped": [c for c in known + past if c not in use_known + use_past],
            "covariates_used": bool(use_known or use_past)}


def context_cap(plan, model_max=None):
    """Context length: the last 2,048 rows at daily origins, all rows for monthly data,
    and never more than the model's own maximum."""
    cap = DAILY_CONTEXT if plan["grain"] != "monthly" else None
    if model_max:
        cap = model_max if cap is None else min(cap, model_max)
    return cap


def build_inputs(frame, rows, plan, cap=None):
    """One SeriesInput per (split row, series) for a channel-independent model, or one per
    split row with a (V, T) target for a joint multivariate model.

    Returns (inputs, keys, tests): keys[i] = (series names, horizon, window, origin date),
    tests[i] = DataFrame of the true target values over the test window.
    """
    inputs, keys, tests = [], [], []
    for _, row in rows.iterrows():
        context, future, test = split_frame(frame, row, past_only=plan["past_only"], known_future=plan["known_future"])
        if cap is not None and len(context) > cap:
            context = context.iloc[-cap:]
        h = int(row["n_test"])
        start = context.index[0]
        known = None
        if plan["known_future"]:
            known = np.vstack([np.r_[context[c].to_numpy(float), future[c].to_numpy(float)]
                               for c in plan["known_future"]])
        past = context[plan["past_only"]].to_numpy(float).T if plan["past_only"] else None
        groups = [plan["targets"]] if plan["joint"] else [[t] for t in plan["targets"]]
        for g in groups:
            y = context[g].to_numpy(float).T
            y = y[0] if len(g) == 1 else y
            item = f"{plan['dataset']}|{'+'.join(g)}|H{h}|r{int(row['window'])}"
            inputs.append(SeriesInput(item, y, plan["freq"], start, past, known,
                                      plan["past_only"] or None, plan["known_future"] or None))
            keys.append((g, h, int(row["window"]), row["origin_date"]))
            tests.append(test[g])
    return inputs, keys, tests


def to_long(model_key, dataset, outputs, keys, tests):
    """Point forecasts in the long layout used by every later notebook:
    dataset, model, series, horizon, window, step, ds, y_true, y_pred."""
    parts = []
    for out, (g, h, r, origin), test in zip(outputs, keys, tests):
        pred = np.atleast_2d(np.asarray(out.mean, float))
        for j, name in enumerate(g):
            parts.append(pd.DataFrame({
                "dataset": dataset, "model": model_key, "series": name, "horizon": h, "window": r,
                "step": np.arange(1, h + 1), "ds": test.index, "y_true": test[name].to_numpy(float),
                "y_pred": pred[j][:h]}))
    return pd.concat(parts, ignore_index=True)
