# Copyright (c) 2026 Shovon Sengupta and Tanujit Chakraborty.
# Companion code for "Hands-On Open-Source Foundation Models for Time Series
# Forecasting" (Packt). Licensed under the MIT License.
"""t0 (The Forecasting Company) and TabPFN-TS (Prior Labs) adapters.

Verified against tfc-t0 0.6.0 and tabpfn-time-series 1.3.0 source:
* ``t0.T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma")``;
  ``.predict(x, horizon, quantile_levels, future_covariates=None)`` with
  ``x`` shaped ``[B, T]`` (independent series) or ``[B, V, T]`` (joint
  multivariate) and ``future_covariates`` ``[B, F, T + horizon]``. Returns a
  ``Forecast`` whose ``.quantiles`` is ``[B, H, Q]`` or ``[B, V, H, Q]``.
  NaN marks missing values. Left-padding a shorter series with NaN inside a batch
  changed its forecast in our check, so series of different lengths go in separate calls.
* ``tabpfn_time_series.TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)``
  (LOCAL is the pipeline default in 1.3.0; the older ``TabPFNTimeSeriesPredictor``
  defaults to the cloud CLIENT). ``predict_df(context_df, future_df, prediction_length,
  quantiles)`` takes columns ``item_id, timestamp, target`` (+ known covariates) and
  returns a frame indexed by (item_id, timestamp) with ``target`` (median) and one
  column per quantile level.
* With tabpfn 9.2.0 the default regressor is ``Prior-Labs/tabpfn_2_5``, which is
  gated: loading it fails without ``TABPFN_TOKEN`` (a free Prior Labs account and
  licence acceptance). The ungated ``Prior-Labs/TabPFN-v2-reg`` weights load without a
  token (both checked on 16 Oct 2026), so this adapter uses them by default.
  Telemetry is switched off with ``TABPFN_DISABLE_TELEMETRY=1``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .base import ForecastOutput, ZeroShotForecaster, select_quantiles


class T0(ZeroShotForecaster):
    """t0-gamma. Targets, past-only covariates and known-future covariates go in as one
    ``t0.TimeSeries`` per input (rows tagged TARGET / HISTORICAL / FUTURE, one group id),
    which is the route that accepts past-only covariates; the array route takes only
    known-future ones. Over the horizon, target and past-only rows are WITHHELD and
    known-future rows carry their values (``t0.rollout.prepare_rollout_buffer``)."""

    name = "t0-gamma"
    family = "t0"
    checkpoint = "theforecastingcompany/t0-gamma"
    license = "Apache-2.0"
    output = "quantiles"
    supports_multivariate = True
    supports_past_covariates = True
    supports_future_covariates = True
    pad_free = True   # NaN left-padding moved the forecast by 0.22 sd (check_models, 16 Oct 2026)

    def _load(self):
        from t0 import T0Forecaster

        self._model = T0Forecaster.from_pretrained(self.checkpoint).to(self.device).eval()

    def _series(self, s, horizon):
        import torch
        from t0.data import TimeSeries
        from t0.types import MaskType, VariateType

        y = np.atleast_2d(np.asarray(s.target, np.float32))               # (V, T)
        V, T = y.shape
        W = T + horizon
        vals, mask, kind = [], [], []

        def add(row, role, known_until):
            v = np.zeros(W, np.float32)
            m = np.full(W, int(MaskType.WITHHELD), np.int8)
            r = np.asarray(row, np.float32)[:known_until]
            v[:len(r)] = np.nan_to_num(r)
            m[:len(r)] = np.where(np.isnan(r), int(MaskType.MISSING), int(MaskType.VALID))
            vals.append(v); mask.append(m); kind.append(int(role))

        for row in y:
            add(row, VariateType.TARGET, T)
        for row in (s.past_covariates if s.past_covariates is not None else []):
            add(row, VariateType.HISTORICAL, T)
        for row in (s.future_covariates if s.future_covariates is not None else []):
            add(row, VariateType.FUTURE, W)
        n = len(vals)
        return TimeSeries(variates=torch.tensor(np.stack(vals)), mask=torch.tensor(np.stack(mask)),
                          group_ids=torch.zeros((n, W), dtype=torch.int64),
                          variate_type=torch.tensor(np.repeat(np.array(kind)[:, None], W, axis=1)))

    def _predict(self, batch, horizon, taus):
        outs = []
        for s in batch:
            fc = self._model.predict(self._series(s, horizon), horizon, quantile_levels=tuple(taus),
                                     context_length=np.asarray(s.target).shape[-1])
            q = fc.quantiles if hasattr(fc, "quantiles") else fc
            q = np.asarray(q.float().cpu() if hasattr(q, "cpu") else q)      # (V, H, Q)
            Q = np.moveaxis(q, -1, 0)                                         # (Q, V, H)
            m = Q[list(taus).index(0.5)] if 0.5 in taus else np.median(Q, axis=0)
            if not s.is_multivariate:
                Q, m = Q[:, 0], m[0]
            outs.append(ForecastOutput(s.item_id, mean=m, quantiles=Q, taus=taus))
        return outs


class TabPFNTS(ZeroShotForecaster):
    name = "TabPFN-TS"
    family = "TabPFN"
    checkpoint = "Prior-Labs/TabPFN-v2-reg"
    license = "tabpfn-time-series: Apache-2.0; weights: Prior Labs licence (check before commercial use)"
    commercial_ok = False
    output = "quantiles"
    supports_future_covariates = True
    # One TabPFN fit per series, one row per context step: TabPFN-v2 was pretrained on up to 10,000 rows,
    # and tabpfn-time-series suggests 4096 as its fast setting (its own default is 32768).
    max_context = 4096

    def __init__(self, checkpoint=None, device="cpu", weights: str = "v2", n_estimators: int | None = None, **kw):
        """weights="v2": ungated TabPFN-v2 regressor (runs without a token).
        weights="default": the library default, ``Prior-Labs/tabpfn_2_5`` with tabpfn 9.2.0,
        which is gated and needs ``TABPFN_TOKEN`` (free Prior Labs account + licence acceptance)."""
        super().__init__(checkpoint=checkpoint, device=device, **kw)
        self.weights = weights
        # TabPFN averages an ensemble of n_estimators passes (library default 8). On a 4-core CPU one
        # 2,048-row daily forecast took about 50 s with 8 and about 25 s with 2 (3 Nov 2026), so the
        # chapter's CPU runs use 2; on a GPU leave it at None (library default).
        self.n_estimators = n_estimators
        if weights == "default" and checkpoint is None:
            self.checkpoint = "Prior-Labs/tabpfn_2_5"

    def _load(self):
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

        cfg = {}
        if self.weights == "v2":
            from tabpfn.model_loading import ModelSource

            cfg = {"model_path": ModelSource.get_regressor_v2().default_filename}
        if self.n_estimators:
            cfg["n_estimators"] = int(self.n_estimators)
        if self.device == "cpu":
            # tabpfn refuses more than 1,000 rows on CPU unless told otherwise (it is slow, not wrong)
            cfg["ignore_pretraining_limits"] = True
        self._model = TabPFNTSPipeline(max_context_length=self.max_context or 32768, tabpfn_mode=TabPFNMode.LOCAL,
                                       tabpfn_model_config=cfg)

    def _predict(self, batch, horizon, taus):
        ctx, fut = [], []
        for i, s in enumerate(batch):
            uid = f"s{i}"  # batch position: several windows of one series may share item_id
            if s.is_multivariate:
                raise ValueError("TabPFN-TS is univariate; split channels first")
            freq = s.freq
            start = pd.Timestamp(s.start) if s.start is not None else pd.Timestamp("2000-01-01")
            T = len(s.target)
            idx = pd.date_range(start, periods=T + horizon, freq=freq)
            c = pd.DataFrame({"item_id": uid, "timestamp": idx[:T], "target": np.asarray(s.target, float)})
            f = pd.DataFrame({"item_id": uid, "timestamp": idx[T:]})
            if s.future_covariates is not None:
                for name, row in zip(s.future_cov_names, s.future_covariates):
                    c[name] = row[:T]
                    f[name] = row[T: T + horizon]
            ctx.append(c)
            fut.append(f)
        # predict_df takes future_df OR prediction_length, not both (ValueError otherwise)
        pred = self._model.predict_df(pd.concat(ctx), future_df=pd.concat(fut), quantiles=list(taus))
        pred = pred.reset_index()
        outs = []
        for i, s in enumerate(batch):
            g = pred[pred["item_id"] == f"s{i}"].sort_values("timestamp")
            cols = [c for c in g.columns if c not in ("item_id", "timestamp", "target")]
            levels = [float(c) for c in cols]
            Q = select_quantiles(np.stack([g[c].to_numpy(float) for c in cols]), levels, taus)
            outs.append(ForecastOutput(s.item_id, mean=g["target"].to_numpy(float), quantiles=Q, taus=taus))
        return outs
