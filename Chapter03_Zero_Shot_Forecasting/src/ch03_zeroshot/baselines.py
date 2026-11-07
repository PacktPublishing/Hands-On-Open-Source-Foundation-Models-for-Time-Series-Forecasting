"""Supervised and statistical baselines (Notebook 10), on exactly the splits the foundation models use.

* Statistical models (naive, seasonal naive, RWD, AR, ARIMA, ARIMAX, ARDL, VAR, VARX) are refitted at
  every forecast origin on the same context a foundation model reads (last 2,048 days for daily data,
  the full history for monthly data). Their orders are chosen at every origin by an information criterion
  on that context only: AIC for AR, ARDL and the VARs, AICc for (S)ARIMA(X) (Hyndman-Khandakar search).
* Learned models (LightGBM, XGBoost, CatBoost, NHiTS, TFT, DLinear, TiDE), all from Darts, are tuned and
  trained once per dataset on the data before the earliest test window, then forecast every window from its
  own origin without retraining (no test data is ever used for fitting or tuning). An Optuna search, whose first
  trial is the default setting, scores each trial on validation windows at the end of the training data; the best
  one is retrained on all training rows. They use a GPU when there is one (LightGBM excepted). A direct model
  reads known-future covariates
  over its whole output, past the end of the shorter test windows: D2's calendar covariates are computed
  for those dates (they are functions of the date).
* Past-only covariates have no future values: ARIMAX and ARDL use plug-in AutoARIMA forecasts of them;
  the learned models read them only through lags (direct multi-step forecasting needs no future values).
* Prices (D3, D4) are modelled in log differences and oil and yields (B1, B2) in first differences by the
  learned models and the VARs; forecasts are integrated back to levels before scoring.
"""
# --- cell: baselines ---
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ch03_zeroshot.data import SEASON, build_inputs, context_cap, input_plan, load_frame, load_manifest, to_long

warnings.filterwarnings("ignore")

# key: name, family, covariates it can use, datasets it runs on (None = all), source
BASELINES = {
    "naive": {"name": "Naive (RW)", "family": "statistical", "exog": "none", "on": None,
              "source": "random walk: last value"},
    "snaive": {"name": "Seasonal naive", "family": "statistical", "exog": "none", "on": None,
               "source": "Hyndman and Athanasopoulos (2021), Sec. 5.2"},
    "rwd": {"name": "RWD", "family": "statistical", "exog": "none", "on": None,
            "source": "random walk with drift; Hyndman and Athanasopoulos (2021), Sec. 5.2"},
    "ar": {"name": "AR", "family": "statistical", "exog": "none", "on": None,
           "source": "statsmodels AutoReg, order by AIC (ar_select_order)"},
    "arima": {"name": "ARIMA", "family": "statistical", "exog": "none", "on": None,
              "source": "Hyndman and Khandakar (2008); statsforecast AutoARIMA"},
    "arimax": {"name": "ARIMAX", "family": "statistical", "exog": "past + known-future", "on": ["D2", "D4"],
               "source": "regression with ARIMA errors; statsforecast AutoARIMA with X"},
    "ardl": {"name": "ARDL", "family": "statistical", "exog": "past + known-future", "on": ["D2", "D4"],
             "source": "Pesaran and Shin (1998); statsmodels ARDL"},
    "var": {"name": "VAR", "family": "statistical", "exog": "none", "on": ["D3", "B2"],
            "source": "Sims (1980); statsmodels VAR, order by AIC"},
    "varx": {"name": "VARX", "family": "statistical", "exog": "past", "on": ["D4"],
             "source": "VAR with the drivers; statsmodels VAR"},
    "lightgbm": {"name": "LightGBM", "family": "machine learning", "exog": "past + known-future", "on": None,
                 "source": "Ke et al. (2017); Darts LightGBMModel"},
    "xgboost": {"name": "XGBoost", "family": "machine learning", "exog": "past + known-future", "on": None,
                "source": "Chen and Guestrin (2016); Darts XGBModel"},
    "catboost": {"name": "CatBoost", "family": "machine learning", "exog": "past + known-future", "on": None,
                 "source": "Prokhorenkova et al. (2018); Darts CatBoostModel"},
    "nhits": {"name": "NHiTS", "family": "deep learning", "exog": "past", "on": None,
              "source": "Challu et al. (2023); Darts NHiTSModel"},
    "tft": {"name": "TFT", "family": "deep learning", "exog": "past + known-future", "on": None,
            "source": "Lim et al. (2021); Darts TFTModel"},
    "dlinear": {"name": "DLinear", "family": "deep learning", "exog": "past + known-future", "on": None,
                "source": "Zeng et al. (2023); Darts DLinearModel"},
    "tide": {"name": "TiDE", "family": "deep learning", "exog": "past + known-future", "on": None,
             "source": "Das et al. (2023); Darts TiDEModel"},
}
STATISTICAL = [k for k, v in BASELINES.items() if v["family"] == "statistical"]
LEARNED = [k for k, v in BASELINES.items() if v["family"] != "statistical"]
TRANSFORM = {"D1": None, "D2": None, "D3": "logdiff", "D4": "logdiff", "B1": "diff", "B2": "diff"}
DRIVER_TRANSFORM = {"global_activity": "diff", "usd": "logdiff", "real_rate_10y": "diff"}
TWIN = {"D2": "D1", "D4": "D3"}
TRAIN_ROWS = 4096          # most recent rows used to train a learned model on daily data
SEED = 42


def caps(key):
    """Capabilities in the form data.input_plan expects."""
    e = BASELINES[key]["exog"]
    return {"multivariate": key in ("var", "varx", "lightgbm", "xgboost", "catboost", "nhits", "tft", "dlinear", "tide"),
            "past_covariates": "past" in e, "future_covariates": "known-future" in e or "future" in e}


def runs_on(key, dataset):
    on = BASELINES[key]["on"]
    return on is None or dataset in on


# ------------------------------------------------------------------ transforms
def _fwd(x, how):
    x = np.asarray(x, float)
    if how is None:
        return x
    if how == "logdiff":
        return np.diff(np.log(x), axis=-1)
    return np.diff(x, axis=-1)


def _inv(last, d, how):
    """Integrate forecasts d (..., H) of a transformed series back to levels from the last level."""
    if how is None:
        return d
    last = np.asarray(last, float)[..., None]
    if how == "logdiff":
        return last * np.exp(np.cumsum(d, axis=-1))
    return last + np.cumsum(d, axis=-1)


# ------------------------------------------------------------------ statistical models
_PLUGIN = {}


def _plugin(name, col, origin, x, h):
    """AutoARIMA forecast of a past-only covariate (cached per origin and horizon)."""
    from statsforecast.models import AutoARIMA

    k = (name, col, str(origin), h)
    if k not in _PLUGIN:
        try:
            _PLUGIN[k] = AutoARIMA(season_length=1).forecast(y=np.asarray(x, float), h=h)["mean"]
        except Exception:
            _PLUGIN[k] = np.repeat(float(x[-1]), h)
    return _PLUGIN[k]


def _exog(name, s, h, lag_past=False):
    """Covariates over the context (T, k) and the horizon (h, k): known-future values are real,
    past-only values are plug-in forecasts. With ``lag_past`` a past-only covariate enters one step late
    (x_{t-1} explains y_t), so step 1 reads its last observed value and only steps 2 and later need plug-ins.
    ARIMAX needs this on D2: its wavelet bands add up exactly to the temperature, so at lag 0 the regression
    would be the identity y_t = sum of the bands_t."""
    cols, ctx, fut = [], [], []
    if s.future_covariates is not None:
        T = len(s.target) if np.ndim(s.target) == 1 else s.target.shape[-1]
        for c, row in zip(s.future_cov_names, s.future_covariates):
            cols.append(c)
            ctx.append(row[:T])
            fut.append(row[T:T + h])
    if s.past_covariates is not None:
        for c, row in zip(s.past_cov_names, s.past_covariates):
            cols.append(c)
            if lag_past:
                ctx.append(np.r_[row[0], row[:-1]])          # first row repeats: no value before the data start
                rest = _plugin(name, c, s.item_id.split("|r")[-1] + str(len(row)), row, h - 1) if h > 1 else []
                fut.append(np.r_[row[-1], rest])
            else:
                ctx.append(row)
                fut.append(_plugin(name, c, s.item_id.split("|r")[-1] + str(len(row)), row, h))
    if not cols:
        return None, None
    return np.column_stack(ctx), np.column_stack(fut)




def _ardl(y, X, Xf, h, maxlag, n_known):
    """ARDL(p, q): known-future covariates enter at lags 0..q, past-only covariates at lags 1..q (their value at
    time t is not known when y_t is forecast). Multi-step forecasts still read plug-in values of the past-only
    covariates for steps 2 and later. The orders p and q are chosen by AIC at every origin, on that origin's
    context only (like AR, ARIMA and VAR), so no window's choice can see another window's test data.
    Returns (forecast, chosen order)."""
    from statsmodels.tsa.ardl import ARDL

    cols = [f"x{i}" for i in range(X.shape[1])]
    X, Xf = pd.DataFrame(X, columns=cols), pd.DataFrame(Xf, columns=cols)

    def order(q):
        return {c: list(range(0 if i < n_known else 1, q + 1)) for i, c in enumerate(cols)}

    best = (np.inf, 1, 1)
    for p in range(1, maxlag + 1):
        for q in (1, 2):
            try:   # hold_back=maxlag: every candidate is fitted on the same rows, so their AICs are comparable
                aic = ARDL(y, p, X, order(q), trend="c", hold_back=maxlag).fit().aic
            except Exception:
                continue
            if aic < best[0]:
                best = (aic, p, q)
    p, q = best[1:]
    res = ARDL(y, p, X, order(q), trend="c").fit()
    return np.asarray(res.forecast(h, exog=Xf), float), f"ARDL({p},{q})"


def _var_forecast(Y, h, maxlags, exog=None, exog_future=None):
    """VAR on stationary columns Y (T, k), lag order p by AIC (1..maxlags); returns ((h, k) forecast, p)."""
    from statsmodels.tsa.api import VAR

    model = VAR(Y, exog=exog)
    p = max(1, int(model.select_order(maxlags).aic))
    res = model.fit(p)
    return res.forecast(Y[-p:], h, exog_future=exog_future), p


def _autoarima(y, h, season, X=None, Xf=None):
    """AutoARIMA (Hyndman and Khandakar, 2008): d by repeated KPSS tests, D by a seasonal-strength test, then
    p, q, P and Q by a stepwise search that minimises the AICc. Returns (forecast, chosen model)."""
    from statsforecast.arima import arima_string
    from statsforecast.models import AutoARIMA

    model = AutoARIMA(season_length=season, ic="aicc").fit(np.asarray(y, float), X=X)
    return np.asarray(model.predict(h, X=Xf)["mean"], float), arima_string(model.model_)


def statistical_forecast(key, s, h, name, m, grain):
    """(point forecast (h,) or (V, h), chosen order) of statistical baseline ``key`` for one SeriesInput.

    Every order is chosen by an information criterion on this origin's context only."""
    y = np.asarray(s.target, float)
    if key == "naive":
        return np.repeat(y[-1], h), ""
    if key == "snaive":
        steps = np.arange(h)
        return y[len(y) - m + steps % m], ""
    if key == "rwd":
        return y[-1] + np.arange(1, h + 1) * (y[-1] - y[0]) / (len(y) - 1), ""
    if key == "ar":
        from statsmodels.tsa.ar_model import AutoReg, ar_select_order

        sel = ar_select_order(y, maxlag=30 if grain != "monthly" else 12, ic="aic", trend="c")
        lags = sel.ar_lags or [1]
        return np.asarray(AutoReg(y, lags=lags, trend="c").fit().forecast(h), float), f"AR({max(lags)})"
    if key in ("arima", "arimax"):
        X, Xf = _exog(name, s, h, lag_past=True) if key == "arimax" else (None, None)
        return _autoarima(y, h, 12 if grain == "monthly" else 1, X, Xf)
    if key == "ardl":
        X, Xf = _exog(name, s, h)
        n_known = 0 if s.future_covariates is None else len(s.future_cov_names)
        return _ardl(y, X, Xf, h, 14 if grain != "monthly" else 12, n_known)
    if key == "var":
        how = TRANSFORM[name]
        d, p = _var_forecast(_fwd(y, how).T, h, 10 if grain != "monthly" else 6)
        return _inv(y[:, -1], d.T, how), f"VAR({p})"
    if key == "varx":
        how = TRANSFORM[name]
        Y = _fwd(y, how).T                                     # (T-1, V)
        # D4: the drivers join the system as extra equations, so they are forecast with the metals (past values only)
        D = np.column_stack([_fwd(r, DRIVER_TRANSFORM[c]) for c, r in zip(s.past_cov_names, s.past_covariates)])
        d, p = _var_forecast(np.hstack([Y, D]), h, 6)
        return _inv(y[:, -1], d[:, :Y.shape[1]].T, how), f"VAR({p}) with the drivers"
    raise KeyError(key)


PARALLEL = ("ar", "arima", "arimax", "ardl")     # refitted at every origin in parallel processes (threads())


def _stat_one(key, s, h, name, m, grain):
    """One origin of a statistical baseline: (forecast, chosen order, error message or None)."""
    import contextlib
    import io

    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f, order = statistical_forecast(key, s, h, name, m, grain)
        return f, order, None
    except Exception as err:
        return np.full(np.shape(s.target)[:-1] + (h,), np.nan), "failed", f"{type(err).__name__}: {err}"


# ------------------------------------------------------------------ learned models (Darts)
TREES = ("lightgbm", "xgboost", "catboost")
VAL_WINDOWS = 2     # validation windows of the longest horizon, cut from the end of the training data


def threads():
    """CPU threads a model may use: all cores, unless CH03_THREADS says otherwise (set it to 1 when several
    runs share the machine; multi-threaded libraries stall badly on an oversubscribed CPU)."""
    return max(1, int(os.environ.get("CH03_THREADS", os.cpu_count() or 1)))


def gpu_available():
    """True when PyTorch sees a CUDA GPU (Colab's T4, for example). The neural networks, XGBoost and CatBoost then
    train on it; LightGBM's pip wheel has no GPU build and stays on the CPU."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def defaults(key, grain):
    """The first setting every search tries: the library's own defaults where they suit a direct forecast of up to
    90 steps (Darts' for the networks, the boosting libraries' for tree size and learning rate)."""
    return {
        "lightgbm": {"num_leaves": 31, "learning_rate": 0.1},
        "xgboost": {"max_depth": 6, "learning_rate": 0.3},
        "catboost": {"depth": 6, "learning_rate": 0.1},
        "nhits": {"layer_widths": 512, "lr": 1e-3},
        "tft": {"hidden_size": 16, "lr": 1e-3},
        "dlinear": {"kernel_size": 25 if grain != "monthly" else 13, "lr": 1e-3},
        "tide": {"hidden_size": 128, "lr": 1e-3},
    }[key]


def search_space(key, grain):
    """The values Optuna may try, a short list per hyperparameter (a size and the learning rate); each list contains
    the default. Everything else is fixed (see ``_settings`` and ``_make``). Widen the lists, or pass your own
    ``space`` to ``run_baseline``, to run a larger search."""
    daily = grain != "monthly"
    return {
        "lightgbm": {"num_leaves": [15, 31, 63], "learning_rate": [0.05, 0.1, 0.2]},
        "xgboost": {"max_depth": [3, 6, 8], "learning_rate": [0.1, 0.3]},
        "catboost": {"depth": [4, 6, 8], "learning_rate": [0.05, 0.1, 0.2]},
        "nhits": {"layer_widths": [128, 256, 512], "lr": [3e-4, 1e-3, 3e-3]},
        "tft": {"hidden_size": [8, 16, 32], "lr": [3e-4, 1e-3, 3e-3]},
        "dlinear": {"kernel_size": [7, 25, 61] if daily else [3, 7, 13], "lr": [3e-4, 1e-3, 3e-3]},
        "tide": {"hidden_size": [32, 64, 128], "lr": [3e-4, 1e-3, 3e-3]},
    }[key]


# Trials per search, trial 0 (the default) included. Sized so that FULL_RUN = True in Notebook 10 stays within one to
# two hours on a CPU: the slow fits (CatBoost, XGBoost, TFT) get fewer trials, the quick ones (DLinear, NHiTS) more.
TRIALS = {"lightgbm": 4, "xgboost": 3, "catboost": 2, "nhits": 5, "tft": 3, "dlinear": 6, "tide": 4}


def suggest(trial, space):
    """One Optuna trial's hyperparameters, each drawn from its list of values."""
    return {name: trial.suggest_categorical(name, list(values)) for name, values in space.items()}


def _settings(name, grain, h_max, smoke):
    daily = grain != "monthly"
    annual = name in ("D1", "D2")       # same days last year (every 3rd day) for temperature
    lags = list(range(-14, 0)) + (list(range(-365, -275, 3)) if annual else []) if daily else list(range(-12, 0))
    return {"in_len": 400 if daily else 36, "tft_in": 120 if daily else 36, "out": h_max, "lags": lags,
            "lags_past": list(range(-7, 0)) if daily else list(range(-6, 0)),
            "lags_future": list(range(0, h_max, 10)) if daily else list(range(0, h_max)),
            "epochs": {"nhits": 10, "tft": 8, "dlinear": 15, "tide": 10}, "trees": 100, "smoke": smoke}


def _ts(values):
    from darts import TimeSeries

    v = np.asarray(values, np.float32)
    return TimeSeries.from_values(v if v.ndim == 2 else v[:, None])


def _make(key, st, have_past, have_future, params=None, use_gpu=False):
    """A fresh Darts model for ``key`` with the dataset's settings and the hyperparameters ``params``."""
    from darts.models import (CatBoostModel, DLinearModel, LightGBMModel, NHiTSModel, TFTModel, TiDEModel,
                              XGBModel)

    p = dict(params or {})
    common = {"output_chunk_length": st["out"], "random_state": SEED}
    if key in TREES:
        trees = 40 if st["smoke"] else st["trees"]
        kw = dict(lags=st["lags"], lags_past_covariates=st["lags_past"] if have_past else None,
                  lags_future_covariates=st["lags_future"] if have_future else None, multi_models=True, **common)
        if key == "lightgbm":     # one ensemble per forecast step (LightGBM has no multi-output trees)
            return LightGBMModel(**{"n_estimators": trees, "verbose": -1, "n_jobs": threads(), **kw, **p})
        if key == "xgboost":      # one booster; each round adds one tree per forecast step and series
            return XGBModel(**{"n_estimators": trees, "tree_method": "hist", "multi_strategy": "one_output_per_tree",
                               "device": "cuda" if use_gpu else "cpu", "n_jobs": threads(), **kw, **p})
        # MultiRMSE: one CatBoost ensemble predicts every step and series at once
        gpu = {"task_type": "GPU", "devices": "0"} if use_gpu else {"thread_count": threads()}
        return CatBoostModel(**{"iterations": trees, "loss_function": "MultiRMSE", "verbose": False, **gpu, **kw, **p})
    trainer = {"enable_progress_bar": False, "enable_model_summary": False, "logger": False,
               "enable_checkpointing": False, "accelerator": "gpu" if use_gpu else "cpu", "devices": 1}
    if "lr" in p:
        p["optimizer_kwargs"] = {"lr": p.pop("lr")}
    for log in ("lightning.pytorch", "pytorch_lightning", "lightning.fabric"):   # no "GPU available" lines
        logging.getLogger(log).setLevel(logging.WARNING)
    epochs = p.pop("n_epochs", st["epochs"][key])
    torch_kw = {"n_epochs": 1 if st["smoke"] else epochs, "batch_size": 64, "pl_trainer_kwargs": trainer, **common}
    if key == "nhits":
        return NHiTSModel(input_chunk_length=st["in_len"], **torch_kw, **p)
    if key == "dlinear":
        return DLinearModel(input_chunk_length=st["in_len"], **torch_kw, **p)
    if key == "tide":
        return TiDEModel(input_chunk_length=st["in_len"], use_static_covariates=False, **torch_kw, **p)
    if key == "tft":
        return TFTModel(**{"input_chunk_length": st["tft_in"], "lstm_layers": 1, "num_attention_heads": 2,
                           "add_relative_index": not have_future, "likelihood": None,
                           "loss_fn": __import__("torch").nn.MSELoss(), **torch_kw, **p})
    raise KeyError(key)


class _Learned:
    """Train once on the pre-test data of a dataset, then forecast each split from its origin."""

    def __init__(self, key, name, frame, manifest, plan, smoke=False, first=None, params=None):
        self.key, self.name, self.plan = key, name, plan
        self.how = TRANSFORM[name]
        self.frame = frame
        self.params = dict(params or {})
        self.st = _settings(name, plan["grain"], int(manifest["horizon"].max()), smoke)
        first = first or manifest["origin_date"].min()     # train on the rows before the earliest test window
        self.train_end = int(np.searchsorted(frame.index, first, side="right"))
        self.start = max(0, self.train_end - TRAIN_ROWS) if plan["grain"] != "monthly" else 0
        self.device = "cpu"

    # transformed arrays for rows [a, b)
    def _target(self, a, b):
        return _fwd(self.frame[self.plan["targets"]].iloc[a:b].to_numpy(float).T, self.how).T

    def _cov(self, cols, a, b):
        if not cols:
            return None
        if self.how is None:
            return self.frame[cols].iloc[a:b].to_numpy(float)
        out = []
        for c in cols:
            x = self.frame[c].iloc[a:b].to_numpy(float)
            out.append(_fwd(x, DRIVER_TRANSFORM.get(c)) if c in DRIVER_TRANSFORM else x[1:])
        return np.column_stack(out)

    def fit(self, end=None):
        """Fit on rows [start, end); end defaults to the end of the training data (the earliest origin). The
        scalers are fitted on the same rows, so nothing after ``end`` is ever seen. Uses the GPU when there is one
        (LightGBM excepted) and falls back to the CPU if the GPU fit fails."""
        from darts.dataprocessing.transformers import Scaler

        a, b = self.start, end or self.train_end
        past, fut = self.plan["past_only"], self.plan["known_future"]
        y = _ts(self._target(a, b))
        self.scaler = Scaler()
        y = self.scaler.fit_transform(y)
        pc = self._cov(past, a, b)
        fc = self._cov(fut, a, b)
        self.pscaler = Scaler() if pc is not None else None
        self.fscaler = Scaler() if fc is not None else None
        kw = {}
        if pc is not None:
            kw["past_covariates"] = self.pscaler.fit_transform(_ts(pc))
        if fc is not None:
            kw["future_covariates"] = self.fscaler.fit_transform(_ts(fc))
        use_gpu = gpu_available() and self.key != "lightgbm"
        try:
            self.model = _make(self.key, self.st, pc is not None, fc is not None, self.params, use_gpu)
            self.model.fit(y, **kw)
            self.device = "gpu" if use_gpu else "cpu"
        except Exception:
            if not use_gpu:
                raise
            self.model = _make(self.key, self.st, pc is not None, fc is not None, self.params, False)
            self.model.fit(y, **kw)
            self.device = "cpu (GPU fit failed)"
        return self

    def predict(self, n_context, h):
        """Forecast h steps from the origin after row n_context; returns (V, h) levels."""
        past, fut = self.plan["past_only"], self.plan["known_future"]
        a = 0 if self.plan["grain"] == "monthly" else max(0, n_context - TRAIN_ROWS)
        y = self.scaler.transform(_ts(self._target(a, n_context)))
        kw = {}
        if self.pscaler is not None:
            kw["past_covariates"] = self.pscaler.transform(_ts(self._cov(past, a, n_context)))
        if self.fscaler is not None:
            kw["future_covariates"] = self.fscaler.transform(_ts(self._cov(fut, a, n_context + self.st["out"])))
        pred = self.model.predict(n=h, series=y, **kw)
        d = self.scaler.inverse_transform(pred).values()[:h].T          # (V, h)
        last = self.frame[self.plan["targets"]].iloc[n_context - 1].to_numpy(float)
        return _inv(last, d, self.how) if self.how else d

    def validation_score(self, val_start, lag):
        """MASE over VAL_WINDOWS windows of the longest horizon that start at row ``val_start`` and end at the end
        of the training data (each forecast from its own origin, without retraining, as in the test protocol).
        The MASE scale is the in-sample naive error (lag ``lag``) of the rows the model was fitted on."""
        h = self.st["out"]
        y = self.frame[self.plan["targets"]].to_numpy(float).T          # (V, N)
        fitted = y[:, self.start:val_start]
        scale = np.mean(np.abs(fitted[:, lag:] - fitted[:, :-lag]), axis=1)
        scores = []
        for origin in range(val_start, self.train_end - h + 1, h):
            pred = np.atleast_2d(self.predict(origin, h))
            scores.append(np.mean(np.abs(pred - y[:, origin:origin + h]), axis=1) / scale)
        return float(np.mean(scores))


def tune(key, name, frame, manifest, plan, first=None, verbose=False, space=None, n_trials=None):
    """Optuna search of one learned baseline on validation windows inside the training data.

    Trial 0 is the default setting (``defaults``); the next ones are drawn by Optuna's TPE sampler from ``space``
    (default ``search_space``; seeded, so the search is repeatable). Each trial is fitted on the training rows before the
    validation windows and scored by its MASE on VAL_WINDOWS windows of the longest horizon at the end of the
    training data. The test windows play no part. Returns the best setting and a record of every trial."""
    import optuna

    from ch03_zeroshot.metrics import SCALE_LAG

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    grain = plan["grain"]
    space = space or search_space(key, grain)
    rows, info = [], {}

    seen = {}

    def objective(trial):
        params = suggest(trial, space)
        key_ = json.dumps(params, sort_keys=True)
        if key_ in seen:                  # Optuna may draw a setting twice: reuse its score, fit nothing
            return seen[key_]
        t0 = time.perf_counter()
        learner = _Learned(key, name, frame, manifest, plan, first=first, params=params)
        val_start = learner.train_end - VAL_WINDOWS * learner.st["out"]
        learner.fit(end=val_start)
        score = learner.validation_score(val_start, SCALE_LAG[name])
        rows.append({"trial": trial.number, "params": params, "val_MASE": round(score, 4),
                     "seconds": round(time.perf_counter() - t0, 1), "device": learner.device})
        info.update(learner=learner, val_start=val_start)
        if verbose:
            print(f"  {key} {name} trial {trial.number} {params}: validation MASE {score:.4f}", flush=True)
        seen[key_] = score
        return score

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=3))
    first_trial = {k: v for k, v in defaults(key, grain).items() if k in space and v in space[k]}
    starts_at_default = len(first_trial) == len(space)
    if first_trial:
        study.enqueue_trial(first_trial)
    wanted = min(n_trials or TRIALS[key], int(np.prod([len(v) for v in space.values()])))

    def enough(study, trial):             # stop after ``wanted`` distinct settings
        if len(rows) >= wanted:
            study.stop()

    study.optimize(objective, n_trials=20 * wanted, callbacks=[enough])
    best = min(rows, key=lambda r: r["val_MASE"])
    learner, val_start = info["learner"], info["val_start"]
    record = {"model": key, "dataset": name, "search": "Optuna TPE (seed 42), trial 0 = defaults",
              "search_space": space,
              "defaults": defaults(key, grain), "trials": rows, "best": best["params"], "best_val_MASE": best["val_MASE"],
              "default_val_MASE": rows[0]["val_MASE"] if starts_at_default else None,
              "validation": {"windows": VAL_WINDOWS, "horizon": learner.st["out"],
                             "first_date": str(frame.index[val_start].date()),
                             "last_date": str(frame.index[learner.train_end - 1].date()),
                             "fitted_from": str(frame.index[learner.start].date()),
                             "earliest_test_origin": str(frame.index[learner.train_end - 1].date())}}
    return best["params"], record


def _label(params):
    return ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())


def tuning_results(root, keys=None, datasets=None):
    """One row per (model, dataset, trial) from the stored searches (results/tuning/), best trial marked."""
    rows = []
    for p in sorted((Path(root) / "results" / "tuning").glob("*/*.json")):
        t = json.loads(p.read_text())
        if (keys and t["model"] not in keys) or (datasets and t["dataset"] not in datasets):
            continue
        for c in t["trials"]:
            rows.append({"model": BASELINES[t["model"]]["name"], "dataset": t["dataset"], "trial": c["trial"],
                         "setting": ("default: " if c["trial"] == 0 and t.get("default_val_MASE") else "")
                         + _label(c["params"]),
                         "validation MASE": c["val_MASE"], "chosen": "yes" if c["params"] == t["best"] else "",
                         "seconds": c["seconds"]})
    return pd.DataFrame(rows)


def tuning_summary(root, keys=None, datasets=None):
    """One row per (model, dataset) of the stored searches: the default and the chosen setting with their
    validation MASE, and how long the search took."""
    rows = []
    for p in sorted((Path(root) / "results" / "tuning").glob("*/*.json")):
        t = json.loads(p.read_text())
        if (keys and t["model"] not in keys) or (datasets and t["dataset"] not in datasets):
            continue
        d = t.get("default_val_MASE")
        rows.append({"model": BASELINES[t["model"]]["name"], "dataset": t["dataset"],
                     "validation MASE (default)": d, "chosen": _label(t["best"]),
                     "validation MASE (chosen)": t["best_val_MASE"],
                     "gain over default %": round(100 * (1 - t["best_val_MASE"] / d), 1) if d else None,
                     "trials": len(t["trials"]), "search seconds": round(sum(c["seconds"] for c in t["trials"]), 1),
                     "device": t["trials"][-1].get("device", "cpu")})
    order = {BASELINES[k]["name"]: i for i, k in enumerate(BASELINES)}
    df = pd.DataFrame(rows)
    return df.sort_values(["model", "dataset"], key=lambda c: c.map(order) if c.name == "model" else c) \
        .set_index(["model", "dataset"]) if len(df) else df


def tuning_design(root, datasets=None):
    """The validation design of each dataset (the same for every learned baseline), read from the stored searches,
    with a check that the validation windows end before the first test window starts."""
    rows = {}
    for p in sorted((Path(root) / "results" / "tuning").glob("*/*.json")):
        t = json.loads(p.read_text())
        if datasets and t["dataset"] not in datasets:
            continue
        v = t["validation"]
        row = {"candidates fitted on": f"{v['fitted_from']} to the day before {v['first_date']}",
               "validation windows": f"{v['windows']} × {v['horizon']} steps, {v['first_date']} to {v['last_date']}",
               "earliest test origin": v["earliest_test_origin"],
               "validation ends by the first origin": v["last_date"] <= v["earliest_test_origin"]}
        if rows.setdefault(t["dataset"], row) != row:
            raise ValueError(f"{t['dataset']}: learned baselines were tuned on different validation windows")
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("dataset")


def chosen_orders(root, datasets=None):
    """The orders the statistical baselines chose at every origin (results/orders/), one row per forecast."""
    parts = [pd.read_csv(p) for p in sorted((Path(root) / "results" / "orders").glob("*/*.csv"))]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    return df[df["dataset"].isin(datasets)] if datasets else df


def order_summary(orders, names=None):
    """How often each order was chosen, per (baseline, dataset, series): the most frequent order and its share."""
    rows = []
    joint = orders["model"].isin(["var", "varx"])          # one system for all series: one order per forecast
    orders = pd.concat([orders[~joint], orders[joint].assign(series="all (one system)")
                        .drop_duplicates(["model", "dataset", "horizon", "window"])])
    for (m, d, s), g in orders.groupby(["model", "dataset", "series"], sort=False):
        counts = g["order"].value_counts()
        rows.append({"baseline": (names or {}).get(m, m), "dataset": d, "series": s, "forecasts": len(g),
                     "most frequent order": counts.index[0], "share": f"{counts.iloc[0] / len(g):.0%}",
                     "distinct orders": len(counts)})
    return pd.DataFrame(rows).set_index(["baseline", "dataset", "series"])


def extend_calendar(frame, cols, steps):
    """``frame`` plus ``steps`` future daily rows in which the calendar covariates ``cols`` (Fourier terms,
    day length) are computed from the date; every other column is NaN. Checked against the stored values."""
    from ch03_dataprep.features import day_length_hours, fourier_terms

    def calendar(idx):
        out = fourier_terms(idx, k_max=3).assign(daylength_h=day_length_hours(idx).to_numpy())
        return out[cols]

    if not np.allclose(calendar(frame.index).to_numpy(), frame[cols].to_numpy(float), atol=1e-6):
        raise ValueError("the calendar covariates do not match the stored dataset")
    idx = pd.date_range(frame.index[-1], periods=steps + 1, freq="D")[1:]
    ext = pd.DataFrame(np.nan, index=idx, columns=frame.columns)
    ext[cols] = calendar(idx).to_numpy()
    return pd.concat([frame, ext])


# ------------------------------------------------------------------ runner
def run_baseline(key, datasets, root, smoke=False, verbose=True, space=None, n_trials=None):
    """Forecast every split of ``datasets`` with baseline ``key``; same outputs as runner.run_model.

    Statistical baselines record the order they chose at every origin in results/orders/<dataset>/<key>.csv.
    Learned baselines: a full run (smoke=False) first searches their hyperparameters with Optuna on validation
    windows inside the training data (``space`` and ``n_trials`` override ``search_space`` and ``TRIALS``) and stores
    the search in results/tuning/<dataset>/<key>.json; a quick check (smoke=True) reuses the stored choice (or the
    defaults when there is none) with a smaller training budget."""
    import contextlib
    import io

    from ch03_zeroshot.models.base import ForecastOutput

    root = Path(root)
    out_dir = root / "results" / ("forecasts_smoke" if smoke else "forecasts")
    log = {"model": key, "name": BASELINES[key]["name"], "family": BASELINES[key]["family"], "datasets": {}}
    for name in datasets:
        if not runs_on(key, name):
            continue
        plan = input_plan(root, name, caps(key))
        target = out_dir / name / f"{key}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = {"covariates_used": plan["covariates_used"], "known_future": plan["known_future"],
                 "past_only": plan["past_only"], "joint": plan["joint"]}
        twin = out_dir / TWIN.get(name, "-") / f"{key}.csv"
        if name in TWIN and not plan["covariates_used"] and twin.exists():
            pd.read_csv(twin).assign(dataset=name).to_csv(target, index=False)
            entry.update(status=f"same input as {TWIN[name]} (no usable covariates): forecasts reused", seconds=0.0)
            log["datasets"][name] = entry
            continue
        frame = load_frame(root, name)
        manifest = load_manifest(root, name)
        first = manifest["origin_date"].min()
        if smoke:
            manifest = manifest[manifest["primary"] & manifest["horizon"].isin(sorted(manifest["horizon"].unique())[:2])]
        t0 = time.perf_counter()
        parts = []
        if key in STATISTICAL:
            plan_run = dict(plan, joint=plan["joint"] and key in ("var", "varx"))
            cap = context_cap(plan_run)
            m = SEASON[name]
            orders = []
            for h, rows in manifest.groupby("horizon"):
                inputs, keys, tests = build_inputs(frame, rows, plan_run, cap)
                outs = []
                jobs = [(key, s, int(h), name, m, plan["grain"]) for s in inputs]
                if key in PARALLEL and threads() > 1 and len(jobs) > 1:     # origins are independent fits
                    from joblib import Parallel, delayed
                    results = Parallel(n_jobs=min(threads(), len(jobs)))(delayed(_stat_one)(*j) for j in jobs)
                else:
                    results = [_stat_one(*j) for j in jobs]
                for s, (g, _, w, origin), (f, order, err) in zip(inputs, keys, results):
                    if err:                             # a failed fit is logged, never silently replaced
                        entry.setdefault("errors", []).append(f"{s.item_id}: {err}"[:200])
                    outs.append(ForecastOutput(s.item_id, mean=np.asarray(f, float)))
                    if order:
                        orders += [{"dataset": name, "model": key, "series": series, "horizon": int(h), "window": w,
                                    "origin": str(pd.Timestamp(origin).date()), "order": order} for series in g]
                parts.append(to_long(key, name, outs, keys, tests))
            if orders and not smoke:
                (root / "results" / "orders" / name).mkdir(parents=True, exist_ok=True)
                pd.DataFrame(orders).to_csv(root / "results" / "orders" / name / f"{key}.csv", index=False)
        else:
            # One model per dataset, with an output as long as the longest horizon. A direct model reads
            # known-future covariates over that whole output: D2's calendar terms are computed past the end of
            # the data (they are functions of the date).
            full = load_manifest(root, name)          # every split: the output is as long as the longest horizon
            if plan["known_future"]:
                frame = extend_calendar(frame, plan["known_future"], int(full["horizon"].max()))
            store = root / "results" / "tuning" / name / f"{key}.json"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if smoke:
                    params = json.loads(store.read_text())["best"] if store.exists() else defaults(key, plan["grain"])
                else:
                    params, info = tune(key, name, frame, full, plan, first=first, space=space, n_trials=n_trials)
                    store.parent.mkdir(parents=True, exist_ok=True)
                    store.write_text(json.dumps(info, indent=1, default=str))
                    entry["tuning_seconds"] = round(time.perf_counter() - t0, 1)
                t1 = time.perf_counter()
                one = _Learned(key, name, frame, full, plan, smoke, first=first, params=params).fit()
            entry["train_seconds"] = round(time.perf_counter() - t1, 1)
            entry["hyperparameters"] = params
            entry["device"] = one.device
            for h, rows in manifest.groupby("horizon"):
                outs, keys, tests = [], [], []
                for _, r in rows.iterrows():
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        f = one.predict(int(r["n_context"]), int(h))
                    test = frame[plan["targets"]].iloc[int(r["n_context"]): int(r["n_context"]) + int(h)]
                    item = f"{name}|{'+'.join(plan['targets'])}|H{int(h)}|r{int(r['window'])}"
                    outs.append(ForecastOutput(item, mean=f if len(plan["targets"]) > 1 else f[0]))
                    keys.append((plan["targets"], int(h), int(r["window"]), r["origin_date"]))
                    tests.append(test)
                parts.append(to_long(key, name, outs, keys, tests))
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(target, index=False)
        sec = time.perf_counter() - t0
        entry.update(status="ok" if not entry.get("errors") else "partial", seconds=round(sec, 1),
                     n_forecast_rows=len(df), nan_forecasts=int(df["y_pred"].isna().sum()))
        log["datasets"][name] = entry
        if verbose:
            print(f"{BASELINES[key]['name']} on {name}: {sec:.1f} s"
                  + (f", {len(entry.get('errors', []))} failed fits" if entry.get("errors") else ""), flush=True)
    log["total_seconds"] = round(sum(d.get("seconds", 0) for d in log["datasets"].values()), 1)
    runs = root / "results" / ("runs_smoke" if smoke else "runs")
    runs.mkdir(parents=True, exist_ok=True)
    p = runs / f"{key}.json"
    with open(runs / f".{key}.lock", "w") as lock:
        try:                          # several processes may run one baseline on different datasets at once
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        except ImportError:           # Windows: no lock (run one process per baseline there)
            pass
        if p.exists():
            old = json.loads(p.read_text())
            old["datasets"].update(log["datasets"])
            log = {**old, **{k: v for k, v in log.items() if k != "datasets"}, "datasets": old["datasets"]}
            log["total_seconds"] = round(sum(d.get("seconds", 0) for d in log["datasets"].values()), 1)
        p.write_text(json.dumps(log, indent=1, default=str))
    return log
