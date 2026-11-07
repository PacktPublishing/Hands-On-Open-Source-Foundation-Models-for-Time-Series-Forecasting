# Copyright (c) 2026 Shovon Sengupta and Tanujit Chakraborty.
# Companion code for "Hands-On Open-Source Foundation Models for Time Series
# Forecasting" (Packt). Licensed under the MIT License.
"""Timestral adapters (Mistral AI, released 15 Oct 2026).

Verified against timestral 1.0.0 (github.com/mistralai/timestral @ a534bdbd24, src/timestral/forecaster.py):
* ``TimestralForecaster.from_pretrained(repo, device="cpu", revision=sha)`` loads ``config.json`` and
  ``model.safetensors`` from the Hub (ungated) or a local folder.
* ``.forecast(context (B, V, T), prediction_length=H, past_covariates=(B, C_p, T),
  known_covariates=(B, C_k, T + H))`` returns a ``Forecast`` whose ``quantiles`` is a torch tensor
  ``(B, V, H, 21)`` at ``quantile_levels`` (0.01, 0.05, 0.10, ..., 0.95, 0.99); the book's nine levels
  0.1 ... 0.9 are among them. The variates of one series are forecast jointly; covariate rows only
  condition the forecast and are never forecast themselves. NaN marks a missing value.
* Deterministic: a monotone quantile head, no sampling. ``symmetric_averaging=True`` (default) also
  forecasts ``-context`` and averages the reflected result; ``clip_nonnegative=True`` (default) clamps
  at 0 every series whose context is entirely non-negative. The adapters keep both defaults (the models
  as shipped); pass ``clip_nonnegative=False`` to turn the clamp off.
* All tensors in one call share V, T and the covariate counts, so the adapter groups the batch by them
  (``pad_free`` makes the base class group by context length first, so no series is padded).
* Licences: inference code Apache-2.0; weights "Modified MIT": MIT plus attribution, and no rights for a
  company (or the user's employer) whose consolidated monthly revenue exceeded USD 20 million in the
  preceding month, derivatives included.
"""
from __future__ import annotations

import numpy as np

from .base import CannotForecast, ForecastOutput, ZeroShotForecaster, select_quantiles


class Timestral(ZeroShotForecaster):
    name = "Timestral-209M"
    family = "Timestral"
    checkpoint = "mistralai/Timestral-209M"
    revision = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"   # Hub commit checked on 21 Oct 2026
    license = "Modified MIT (weights; no rights above USD 20M monthly revenue); Apache-2.0 (code)"
    commercial_ok = False   # conditional: see licence (revenue cap)
    output = "quantiles"
    supports_multivariate = True
    supports_past_covariates = True
    supports_future_covariates = True
    pad_free = True

    def __init__(self, checkpoint=None, device="cpu", revision=None, forecaster=None,
                 symmetric_averaging=True, clip_nonnegative=True, **kw):
        super().__init__(checkpoint=checkpoint, device=device, **kw)
        if checkpoint and revision is None:
            self.revision = None            # another checkpoint: its latest Hub commit
        elif revision is not None:
            self.revision = revision
        self._injected = forecaster
        self.symmetric_averaging = symmetric_averaging
        self.clip_nonnegative = clip_nonnegative

    def _load(self):
        if self._injected is not None:
            self._model = self._injected
            return
        from timestral import TimestralForecaster

        self._model = TimestralForecaster.from_pretrained(self.checkpoint, device=self.device,
                                                          revision=self.revision)

    def _predict(self, batch, horizon, taus):
        groups: dict[tuple, list[int]] = {}
        for i, s in enumerate(batch):
            y = np.atleast_2d(np.asarray(s.target))
            kp = 0 if s.past_covariates is None else np.asarray(s.past_covariates).shape[0]
            kf = 0 if s.future_covariates is None else np.asarray(s.future_covariates).shape[0]
            groups.setdefault((y.shape, kp, kf), []).append(i)
        if any(kf for (_, _, kf) in groups) and not self._model.supports_known_covariates:
            raise CannotForecast(f"{self.checkpoint} cannot condition on known-future covariates")
        outs = [None] * len(batch)
        for (shape, kp, kf), idx in groups.items():
            T = shape[-1]
            ctx = np.stack([np.atleast_2d(np.asarray(batch[i].target, np.float32)) for i in idx])   # (B, V, T)
            past = np.stack([np.asarray(batch[i].past_covariates, np.float32) for i in idx]) if kp else None
            fut = np.stack([np.asarray(batch[i].future_covariates, np.float32)[:, :T + horizon]
                            for i in idx]) if kf else None
            f = self._model.forecast(ctx, prediction_length=horizon, past_covariates=past, known_covariates=fut,
                                     symmetric_averaging=self.symmetric_averaging,
                                     clip_nonnegative=self.clip_nonnegative)
            native = np.moveaxis(f.quantiles.float().cpu().numpy(), -1, 0)     # (21, B, V, H)
            median = f.median.float().cpu().numpy()                             # (B, V, H)
            for j, i in enumerate(idx):
                Q = select_quantiles(native[:, j], f.quantile_levels, taus)      # (K, V, H)
                m = median[j]
                if not batch[i].is_multivariate:
                    Q, m = Q[:, 0], m[0]
                outs[i] = ForecastOutput(batch[i].item_id, mean=m, quantiles=Q, taus=taus)
        return outs


class TimestralSmall(Timestral):
    name = "Timestral-5M"
    checkpoint = "mistralai/Timestral-5M"
    revision = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"    # Hub commit checked on 21 Oct 2026
