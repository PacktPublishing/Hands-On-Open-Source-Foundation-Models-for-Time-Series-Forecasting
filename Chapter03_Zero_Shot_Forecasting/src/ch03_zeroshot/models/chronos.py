# Copyright (c) 2026 Shovon Sengupta and Tanujit Chakraborty.
# Companion code for "Hands-On Open-Source Foundation Models for Time Series
# Forecasting" (Packt). Licensed under the MIT License.
"""Chronos-2 and Chronos-Bolt adapters (Amazon, ``chronos-forecasting>=2.0``, Apache-2.0).

API verified against chronos-forecasting 2.3.2 source:
* ``Chronos2Pipeline.predict_quantiles(inputs, prediction_length, quantile_levels)``
  returns ``(quantiles, mean)``: lists with one tensor per input, shaped
  ``(n_variates, H, Q)`` and ``(n_variates, H)``. ``mean`` is the median.
* Inputs may be dicts ``{"target", "past_covariates", "future_covariates"}``;
  every key in ``future_covariates`` must also be in ``past_covariates``
  (its historical values), and all dicts in one call must share a schema.
* ``ChronosBoltPipeline.predict_quantiles(list_of_1d_tensors, prediction_length,
  quantile_levels)`` returns ``(quantiles (B, H, Q), mean (B, H))``.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastOutput, ZeroShotForecaster


def _device_map(device: str) -> str:
    return "cuda" if device.startswith("cuda") else ("mps" if device == "mps" else "cpu")


class Chronos2(ZeroShotForecaster):
    name = "Chronos-2"
    family = "Chronos"
    checkpoint = "amazon/chronos-2"            # 120M; "autogluon/chronos-2-small" (28M) for CPU
    license = "Apache-2.0"
    output = "quantiles"
    supports_multivariate = True
    supports_past_covariates = True
    supports_future_covariates = True
    max_context = 8192

    def _load(self):
        import torch
        from chronos import Chronos2Pipeline

        self._model = Chronos2Pipeline.from_pretrained(
            self.checkpoint, device_map=_device_map(self.device),
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32)

    def _to_input(self, s, horizon):
        y = np.asarray(s.target, np.float32)
        if s.past_covariates is None and s.future_covariates is None:
            return {"target": y}
        past, fut = {}, {}
        T = y.shape[-1]
        if s.past_covariates is not None:
            for name, row in zip(s.past_cov_names, s.past_covariates):
                past[name] = np.asarray(row, np.float32)
        if s.future_covariates is not None:
            for name, row in zip(s.future_cov_names, s.future_covariates):
                past[name] = np.asarray(row[:T], np.float32)
                fut[name] = np.asarray(row[T: T + horizon], np.float32)
        d = {"target": y, "past_covariates": past}
        if fut:
            d["future_covariates"] = fut
        return d

    def _predict(self, batch, horizon, taus):
        inputs = [self._to_input(s, horizon) for s in batch]
        q_list, mean_list = self._model.predict_quantiles(
            inputs, prediction_length=horizon, quantile_levels=list(taus),
            batch_size=self.kwargs.get("batch_size", 256),
            cross_learning=self.kwargs.get("cross_learning", False))
        outs = []
        for s, q, mu in zip(batch, q_list, mean_list):
            q = q.float().cpu().numpy()          # (V, H, Q)
            mu = mu.float().cpu().numpy()        # (V, H)
            Q = np.moveaxis(q, -1, 0)            # (Q, V, H)
            if not s.is_multivariate:
                Q, mu = Q[:, 0], mu[0]
            outs.append(ForecastOutput(s.item_id, mean=mu, quantiles=Q, taus=taus))
        return outs


class ChronosBolt(ZeroShotForecaster):
    name = "Chronos-Bolt"
    family = "Chronos"
    checkpoint = "amazon/chronos-bolt-small"   # tiny 9M / mini 21M / small 48M / base 205M
    license = "Apache-2.0"
    output = "quantiles"
    max_context = 2048

    def _load(self):
        from chronos import BaseChronosPipeline

        self._model = BaseChronosPipeline.from_pretrained(self.checkpoint, device_map=_device_map(self.device))

    def _predict(self, batch, horizon, taus):
        import torch

        flat, owners = [], []
        for i, s in enumerate(batch):
            y = np.asarray(s.target, np.float32)
            rows = [y] if y.ndim == 1 else list(y)          # channel-independent
            flat += [torch.tensor(r) for r in rows]
            owners += [i] * len(rows)
        q, mu = self._model.predict_quantiles(flat, prediction_length=horizon, quantile_levels=list(taus))
        q = q.float().cpu().numpy()       # (N, H, Q)
        mu = mu.float().cpu().numpy()     # (N, H)
        owners = np.array(owners)
        outs = []
        for i, s in enumerate(batch):
            idx = np.where(owners == i)[0]
            Q = np.moveaxis(q[idx], -1, 0)          # (Q, V, H)
            m = mu[idx]
            if not s.is_multivariate:
                Q, m = Q[:, 0], m[0]
            outs.append(ForecastOutput(s.item_id, mean=m, quantiles=Q, taus=taus))
        return outs
