# Copyright (c) 2026 Shovon Sengupta and Tanujit Chakraborty.
# Companion code for "Hands-On Open-Source Foundation Models for Time Series
# Forecasting" (Packt). Licensed under the MIT License.
"""Toto 2.0 (``toto-2``, import ``toto2``) and Toto 1.0 (``toto-ts``, import ``toto``).

Both need Python >= 3.12. They live in separate environments (``env-toto2``
and ``env-toto1``) because ``toto-ts`` pins torch==2.7.0 and numpy 1.26.

Verified against toto-2 2.0.0 and toto-ts 0.2.0 source:

* Toto 2.0: ``Toto2Model.from_pretrained("Datadog/Toto-2.0-22m")`` then
  ``model.forecast({"target": (B, V, T), "target_mask": bool (B, V, T),
  "series_ids": long (B, V)}, horizon=H, decode_block_size=None)`` returns the
  nine deciles with the quantile axis first: ``(9, B, V, H)``
  (``toto2/model.py``). The README says exogenous variables are "planned for a
  future 2.0 release", so this adapter does not pass covariates to Toto 2.0.
* Toto 1.0: ``Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")``;
  ``TotoForecaster(toto.model).forecast(MaskedTimeseries(...), prediction_length=H,
  num_samples=S, future_exogenous_variables=(B, K, H))``. Known-future
  covariates must be the *last* K variates of ``series`` and are passed with
  ``num_exogenous_variables=K``; the output still contains those channels, so
  they are sliced off. The README takes the median of 256 samples as the point
  forecast.
* Left padding: series of different lengths are left-padded and the padding is
  marked invalid in the mask (``target_mask`` / ``padding_mask``). Toto 2.0 also
  needs the context to be a multiple of its patch size; the oldest points are
  dropped to get there, as ``Toto2GluonTSModel`` does (``toto2/model.py:1465``).
"""
from __future__ import annotations

import numpy as np

from .base import ForecastOutput, ZeroShotForecaster, quantiles_from_samples, select_quantiles

_TOTO2_TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

def _interval_seconds(freq: str) -> int:
    import pandas as pd

    try:
        return int(pd.Timedelta(pd.tseries.frequencies.to_offset(freq)).total_seconds())
    except (ValueError, TypeError):  # month / quarter offsets have no fixed length
        return 30 * 86400


def _left_pad(arrs, T):
    """Stack (V, T_i) arrays into (B, V, T) with left padding and a validity mask."""
    B, V = len(arrs), arrs[0].shape[0]
    x = np.zeros((B, V, T), np.float32)
    mask = np.zeros((B, V, T), bool)
    for b, a in enumerate(arrs):
        t = a.shape[-1]
        x[b, :, T - t:] = np.nan_to_num(a)
        mask[b, :, T - t:] = ~np.isnan(a)
    return x, mask


class Toto2(ZeroShotForecaster):
    name = "Toto-2.0"
    family = "Toto"
    checkpoint = "Datadog/Toto-2.0-22m"
    license = "Apache-2.0"
    output = "quantiles"
    supports_multivariate = True
    max_context = 4096            # harness default; the model accepts variable context lengths

    def __init__(self, checkpoint=None, device="cpu", model=None, decode_block_size=None, **kw):
        super().__init__(checkpoint=checkpoint, device=device, **kw)
        self._injected = model
        self.decode_block_size = decode_block_size

    def _load(self):
        from toto2 import Toto2Model

        model = self._injected if self._injected is not None else Toto2Model.from_pretrained(self.checkpoint)
        self._model = model.to(self.device).eval()

    def _predict(self, batch, horizon, taus):
        import torch

        groups: dict[int, list[int]] = {}
        for i, s in enumerate(batch):
            groups.setdefault(np.atleast_2d(s.target).shape[0], []).append(i)
        outs = [None] * len(batch)
        for V, idx in groups.items():
            # The context must be a whole number of patches. Like Toto2GluonTSModel,
            # drop the oldest points down to a multiple of the patch size, and
            # left-pad (masked) anything shorter than the longest series.
            p = self._model.config.patch_size
            arrs = []
            for i in idx:
                a = np.atleast_2d(np.asarray(batch[i].target, float))
                keep = max(p, a.shape[-1] // p * p)
                arrs.append(a[:, -keep:])
            T = max(a.shape[-1] for a in arrs)
            x, mask = _left_pad(arrs, T)
            inputs = {"target": torch.tensor(x, device=self.device),
                      "target_mask": torch.tensor(mask, device=self.device),
                      "series_ids": torch.zeros(len(idx), V, dtype=torch.long, device=self.device)}
            q = self._model.forecast(inputs, horizon=horizon, decode_block_size=self.decode_block_size,
                                     has_missing_values=bool((~mask).any()))
            q = q.float().cpu().numpy()                       # (9, B, V, H)
            for j, i in enumerate(idx):
                Q = select_quantiles(q[:, j], _TOTO2_TAUS, taus)   # (K, V, H)
                med = q[4, j]
                if not batch[i].is_multivariate:
                    Q, med = Q[:, 0], med[0]
                outs[i] = ForecastOutput(batch[i].item_id, mean=med, quantiles=Q, taus=taus)
        return outs


class Toto1(ZeroShotForecaster):
    """Toto 1.0, 256 sample paths per forecast (as in the README).

    ``TotoForecaster.forecast`` decodes ``batch x samples_per_batch`` sequences at once. One ETTh1 series
    at a 4,096-step context with all 256 samples in one batch took about 31 s on a 4-core CPU; three
    electricity series in one call passed 13.6 GB and were killed on a 15 GB machine (16 Oct 2026).
    Calls are therefore split so that series x samples per batch x variates x context steps stays below
    ``budget`` (that ETTh1 call is exactly at it): fewer series per call first, then smaller sample batches
    (a divisor of ``num_samples``, as toto requires). All ``num_samples`` samples are still drawn.
    """

    name = "Toto-1.0"
    family = "Toto"
    checkpoint = "Datadog/Toto-Open-Base-1.0"
    license = "Apache-2.0"
    output = "samples"
    supports_multivariate = True
    supports_future_covariates = True
    max_context = 4096            # the README example uses 4096 steps
    budget = 256 * 4096           # series x samples per batch x variates x context steps per call

    def __init__(self, checkpoint=None, device="cpu", num_samples: int = 256, samples_per_batch: int = 256,
                 model=None, **kw):
        super().__init__(checkpoint=checkpoint, device=device, **kw)
        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch
        self._injected = model

    def _load(self):
        from toto.inference.forecaster import TotoForecaster
        from toto.model.toto import Toto

        toto = self._injected if self._injected is not None else Toto.from_pretrained(self.checkpoint)
        toto = toto.to(self.device).eval()
        self._model = TotoForecaster(toto.model)

    def _calls(self, idx, cost):
        """Split one group into ``(series, samples_per_batch)`` calls that stay under the budget."""
        spb = min(self.samples_per_batch, self.num_samples)
        if len(idx) * spb * cost <= self.budget:
            return [(idx, spb)]
        if spb * cost <= self.budget:
            n = max(1, int(self.budget // (spb * cost)))
            return [(idx[k: k + n], spb) for k in range(0, len(idx), n)]
        fit = max(1, int(self.budget // cost))
        spb = max(d for d in range(1, spb + 1) if self.num_samples % d == 0 and d <= fit)
        return [([i], spb) for i in idx]

    def _predict(self, batch, horizon, taus):
        groups: dict[tuple, list[int]] = {}
        for i, s in enumerate(batch):
            V = np.atleast_2d(s.target).shape[0]
            K = 0 if s.future_covariates is None else len(s.future_covariates)
            groups.setdefault((V, K), []).append(i)
        outs = [None] * len(batch)
        for (V, K), idx in groups.items():
            T = max(np.asarray(batch[i].target).shape[-1] for i in idx)
            for part, spb in self._calls(idx, (V + K) * T):
                self._predict_call(batch, part, V, K, spb, horizon, taus, outs)
        return outs

    def _predict_call(self, batch, idx, V, K, spb, horizon, taus, outs):
        import torch
        from toto.data.util.dataset import MaskedTimeseries

        arrs, futs = [], []
        for i in idx:
            s = batch[i]
            y = np.atleast_2d(np.asarray(s.target, float))
            T_i = y.shape[-1]
            if K:
                fc = np.asarray(s.future_covariates, float)
                y = np.vstack([y, fc[:, :T_i]])           # exogenous = last K variates
                futs.append(fc[:, T_i: T_i + horizon])
            arrs.append(y)
        T = max(a.shape[-1] for a in arrs)
        x, mask = _left_pad(arrs, T)
        dt = _interval_seconds(batch[idx[0]].freq)
        series = torch.tensor(x, device=self.device)
        inputs = MaskedTimeseries(
            series=series, padding_mask=torch.tensor(mask, device=self.device),
            id_mask=torch.zeros_like(series), timestamp_seconds=torch.zeros_like(series),
            time_interval_seconds=torch.full((len(idx), V + K), dt, device=self.device),
            num_exogenous_variables=K)
        fut = torch.tensor(np.stack(futs), dtype=torch.float32, device=self.device) if K else None
        fc = self._model.forecast(inputs, prediction_length=horizon, num_samples=self.num_samples,
                                  samples_per_batch=spb, future_exogenous_variables=fut)
        S = fc.samples.float().cpu().numpy()[:, :V]          # (B, V, H, S): drop exogenous channels
        for j, i in enumerate(idx):
            smp = np.moveaxis(S[j], -1, 0)                   # (S, V, H)
            if not batch[i].is_multivariate:
                smp = smp[:, 0]
            outs[i] = ForecastOutput(batch[i].item_id, mean=np.median(smp, axis=0),
                                     quantiles=quantiles_from_samples(smp, taus), taus=taus, samples=smp)
