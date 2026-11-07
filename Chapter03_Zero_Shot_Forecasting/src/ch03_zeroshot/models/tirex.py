# Copyright (c) 2026 Shovon Sengupta and Tanujit Chakraborty.
# Companion code for "Hands-On Open-Source Foundation Models for Time Series
# Forecasting" (Packt). Licensed under the MIT License.
"""TiRex (v1) and TiRex-2 adapters (NXAI).

Verified against tirex-ts 1.4.2 and tirex-2 0.3.1 source:
* ``tirex.load_model(path, device=None, backend="torch")`` then
  ``.forecast(context, prediction_length=H, output_type="numpy")`` returns
  ``(quantiles (B, H, 9), mean (B, H))``.
* ``tirex2.load_model(ckpt_path="NX-AI/TiRex-2", device="cuda")`` - note the
  default device is CUDA, so pass ``device="cpu"`` or ``"mps"`` explicitly.
  ``.forecast([TimeseriesType(target (V_t, T), past_covariates (V_p, T) | None,
  future_covariates (V_f, >=T+H) | None)], prediction_length=H, output_type="numpy")``
  returns one array per series shaped ``(V_t, 9, H)``.
* Licences: TiRex v1 weights use the NXAI Community License (a commercial licence
  is needed above EUR 100M revenue when used in a product); TiRex-2 is Apache-2.0.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastOutput, ZeroShotForecaster, select_quantiles

_TIREX_TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class TiRex(ZeroShotForecaster):
    name = "TiRex"
    family = "TiRex"
    checkpoint = "NX-AI/TiRex"
    license = "NXAI Community License"
    commercial_ok = False   # conditional: see licence (> EUR 100M revenue + commercial product)
    output = "quantiles"

    def _load(self):
        from tirex import load_model

        self._model = load_model(self.checkpoint, device=self.device, backend="torch")

    def _predict(self, batch, horizon, taus):
        import torch

        flat, owners = [], []
        for i, s in enumerate(batch):
            y = np.asarray(s.target, np.float32)
            rows = [y] if y.ndim == 1 else list(y)
            flat += [torch.tensor(r) for r in rows]
            owners += [i] * len(rows)
        q, mu = self._model.forecast(context=flat, prediction_length=horizon, output_type="numpy")
        q, mu = np.asarray(q), np.asarray(mu)                 # (N, H, 9), (N, H)
        native = np.moveaxis(q, -1, 0)                         # (9, N, H)
        owners = np.array(owners)
        outs = []
        for i, s in enumerate(batch):
            idx = np.where(owners == i)[0]
            Q = select_quantiles(native[:, idx], _TIREX_TAUS, taus)
            m = mu[idx]
            if not s.is_multivariate:
                Q, m = Q[:, 0], m[0]
            outs.append(ForecastOutput(s.item_id, mean=m, quantiles=Q, taus=taus))
        return outs


class TiRex2(ZeroShotForecaster):
    name = "TiRex-2"
    family = "TiRex"
    checkpoint = "NX-AI/TiRex-2"
    license = "Apache-2.0"
    output = "quantiles"
    # tirex-2 0.3.1: "prediction_length=720 exceeds the supported maximum of 320. It will be
    # truncated to 320." (checked 16 Oct 2026). Longer horizons are skipped, not padded.
    max_horizon = 320
    supports_multivariate = True
    supports_past_covariates = True
    supports_future_covariates = True

    def _load(self):
        from tirex2 import load_model

        self._model = load_model(self.checkpoint, device=self.device)

    def _predict(self, batch, horizon, taus):
        import torch
        from tirex2 import TimeseriesType

        def t(x):
            return None if x is None else torch.as_tensor(np.atleast_2d(np.asarray(x, np.float32)))

        series = [TimeseriesType(target=t(s.target), past_covariates=t(s.past_covariates),
                                 future_covariates=t(s.future_covariates)) for s in batch]
        res = self._model.forecast(series, prediction_length=horizon, output_type="numpy")
        outs = []
        for s, r in zip(batch, res):
            r = np.asarray(r, float)                         # (V, 9, H)
            Q = select_quantiles(np.moveaxis(r, 1, 0), _TIREX_TAUS, taus)   # (K, V, H)
            m = Q[list(taus).index(0.5)] if 0.5 in taus else np.median(Q, axis=0)
            if not s.is_multivariate:
                Q, m = Q[:, 0], m[0]
            outs.append(ForecastOutput(s.item_id, mean=m, quantiles=Q, taus=taus))
        return outs
