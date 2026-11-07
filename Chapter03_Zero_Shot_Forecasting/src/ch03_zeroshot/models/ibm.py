# Copyright (c) 2026 Shovon Sengupta and Tanujit Chakraborty.
# Companion code for "Hands-On Open-Source Foundation Models for Time Series
# Forecasting" (Packt). Licensed under the MIT License.
"""IBM Granite time series adapters: TTM r2, FlowState r1.1, PatchTST-FM r2 (``granite-tsfm``).

Verified against granite-tsfm 0.3.10 (signatures from source; output shapes by
running the model classes with random weights, see tests/test_adapter_shapes.py):
* TTM: ``tsfm_public.toolkit.get_model.get_model(repo, context_length, prediction_length, freq=...)``
  returns a ``TinyTimeMixerForPrediction``; ``model(past_values=(B, T, C)).prediction_outputs``
  is ``(B, H_model, C)``. Point forecasts only. The model must receive exactly
  ``config.context_length`` steps; a shorter context is left-padded and the padded steps are
  marked unobserved (the zero-padding route, see ``TTM._predict``).
* FlowState: ``FlowStateForPrediction.from_pretrained(repo, revision="r1.1")``;
  ``model(past_values=(B, T, 1), prediction_length=H, batch_first=True, scale_factor=s)``.
  Only ONE channel per call. ``prediction_outputs`` is ``(B, H, 1)`` and
  ``quantile_outputs`` is ``(B, 9, H, 1)`` for q0.1..q0.9. ``scale_factor`` = 24 / seasonal period
  per the model card (1.0 for hourly data with daily seasonality). A shorter series that
  is left-padded and masked in a batch does not get the same forecast as alone, so the
  adapter forecasts each context length separately (``pad_free``).
* PatchTST-FM: ``PatchTSTFMForPrediction.from_pretrained(repo)``;
  ``model(past_values=(B, T, 1), prediction_length=H, quantile_levels=[...])`` returns
  ``prediction_outputs (B, H, 1)`` and ``quantile_outputs (B, K, H, 1)``. The model
  is trained on 99 quantile levels 0.01..0.99. (Its docstring says (B, F, H) for
  ``prediction_outputs``; running it gives (B, H, F).) ``prediction_outputs`` is the
  mean (the quantile function integrated over the 99 levels), not the median. A tensor
  input takes the fast path (``forecast_single_step_fast``); a list of variable-length
  tensors takes a per-series path that was about 150x slower on CPU (102 s against
  0.7 s for one 512-step series, 16 Oct 2026), so the adapter batches by context length.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastOutput, ZeroShotForecaster


def _flat(batch):
    flat, owners = [], []
    for i, s in enumerate(batch):
        y = np.asarray(s.target, np.float32)
        rows = [y] if y.ndim == 1 else list(y)
        flat += rows
        owners += [i] * len(rows)
    return flat, np.array(owners)


def _regroup(batch, owners, Q, point, taus):
    outs = []
    for i, s in enumerate(batch):
        idx = np.where(owners == i)[0]
        q = None if Q is None else Q[:, idx]
        m = point[idx]
        if not s.is_multivariate:
            q = None if q is None else q[:, 0]
            m = m[0]
        outs.append(ForecastOutput(s.item_id, mean=m, quantiles=q, taus=taus))
    return outs


def _left_pad_nan(rows):
    """Stack variable-length series into (N, T_max), left-padding with NaN."""
    T = max(len(r) for r in rows)
    out = np.full((len(rows), T), np.nan, np.float32)
    for k, r in enumerate(rows):
        out[k, T - len(r):] = r
    return out


class TTM(ZeroShotForecaster):
    name = "TTM-r2"
    family = "IBM Granite"
    checkpoint = "ibm-granite/granite-timeseries-ttm-r2"
    license = "Apache-2.0"
    output = "point"

    def __init__(self, checkpoint=None, device="cpu", context_length=512, freq=None, **kw):
        super().__init__(checkpoint, device, **kw)
        self.context_length = context_length
        # Shorter contexts are left-padded and masked (IBM's "zeropad" route, see _predict),
        # so no window is skipped; D3 has 505 to 532 months before its later origins.
        self.min_context = None
        self.max_context = context_length
        self.freq = freq
        self._loaded_for = None

    def _load(self):
        self._model = "lazy"   # the branch depends on the horizon; loaded in _predict

    def _get(self, horizon):
        if self._loaded_for == horizon:
            return self._net
        from tsfm_public.toolkit.get_model import get_model

        self._net = get_model(self.checkpoint, context_length=self.context_length,
                              prediction_length=horizon, freq=self.freq).to(self.device).eval()
        self._loaded_for = horizon
        return self._net

    def _predict(self, batch, horizon, taus):
        import torch

        net = self._get(horizon)
        L = net.config.context_length
        flat, owners = _flat(batch)
        # TTM reads exactly L steps. A shorter context is left-padded with NaN, which becomes 0
        # after scaling, and the padded steps are marked unobserved so the model's own scaler
        # ignores them (the zero-padding route of tsfm_public's get_model(force_return="zeropad")).
        X = _left_pad_nan([r[-L:] for r in flat] + [np.full(L, np.nan, np.float32)])[:-1]  # (N, L)
        obs = ~np.isnan(X)
        mu, sd = np.nanmean(X, 1, keepdims=True), np.nanstd(X, 1, keepdims=True) + 1e-8
        Xs = np.nan_to_num((X - mu) / sd)
        kw = {"past_observed_mask": torch.tensor(obs[:, :, None], dtype=torch.float32, device=self.device)}
        if getattr(net.config, "resolution_prefix_tuning", False):
            # r2 branches with a frequency prefix fail without freq_token (checked on the
            # real ttm-r2 checkpoint, 16 Oct 2026); unknown frequencies map to "oov".
            from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING

            f = batch[0].freq
            tok = DEFAULT_FREQUENCY_MAPPING.get(f, DEFAULT_FREQUENCY_MAPPING.get(f.lower(), 0))
            kw["freq_token"] = torch.full((len(flat),), tok, dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = net(past_values=torch.tensor(Xs[:, :, None], device=self.device), **kw).prediction_outputs
        P = out[:, :horizon, 0].float().cpu().numpy() * sd + mu    # (N, H)
        if P.shape[1] < horizon:
            raise ValueError(f"checkpoint branch predicts {P.shape[1]} < {horizon} steps")
        return _regroup(batch, owners, None, P, taus)


class FlowState(ZeroShotForecaster):
    name = "FlowState-r1.1"
    family = "IBM Granite"
    checkpoint = "ibm-granite/granite-timeseries-flowstate-r1"
    revision = "r1.1"
    license = "Apache-2.0"
    output = "quantiles"
    max_context = 4096
    pad_free = True   # with NaN left-padding + pad_mask the forecast moved by 0.34 sd (check_models, 16 Oct 2026)

    def __init__(self, checkpoint=None, device="cpu", season_length=24, **kw):
        super().__init__(checkpoint, device, **kw)
        self.set_season_length(season_length)

    def set_season_length(self, m):
        self.season_length = m
        self.scale_factor = 24.0 / max(m, 1) if m > 1 else 1.0

    def _load(self):
        from tsfm_public import FlowStateForPrediction

        self._model = FlowStateForPrediction.from_pretrained(self.checkpoint, revision=self.revision)
        self._model = self._model.to(self.device).eval()

    def _predict(self, batch, horizon, taus):
        import torch

        from .base import select_quantiles

        flat, owners = _flat(batch)
        X = _left_pad_nan(flat)
        pad = np.isnan(X)
        with torch.no_grad():
            o = self._model(past_values=torch.tensor(np.nan_to_num(X)[:, :, None], device=self.device),
                            pad_mask=torch.tensor(pad, device=self.device) if pad.any() else None,
                            prediction_length=horizon, batch_first=True, scale_factor=self.scale_factor)
        q = o.quantile_outputs[..., 0].float().cpu().numpy()       # (N, 9, H)
        Q = select_quantiles(np.moveaxis(q, 1, 0), self._model.config.quantiles, taus)   # (K, N, H)
        P = o.prediction_outputs[..., 0].float().cpu().numpy()       # (N, H)
        return _regroup(batch, owners, Q, P, taus)


class PatchTSTFM(ZeroShotForecaster):
    name = "PatchTST-FM-r2"
    family = "IBM Granite"
    checkpoint = "ibm-granite/granite-timeseries-patchtst-fm-r2"
    license = "Apache-2.0 / OpenMDW-1.0"
    output = "quantiles"
    max_context = 8192
    pad_free = True   # equal-length batches go through the fast tensor path

    def _load(self):
        from tsfm_public import PatchTSTFMForPrediction

        self._model = PatchTSTFMForPrediction.from_pretrained(self.checkpoint).to(self.device).eval()

    def _predict(self, batch, horizon, taus):
        import torch

        flat, owners = _flat(batch)
        X = torch.tensor(np.stack(flat)[:, :, None], device=self.device)     # (N, T, 1), one length
        with torch.no_grad():
            o = self._model(past_values=X, prediction_length=horizon, quantile_levels=list(taus))
        q = o.quantile_outputs[..., 0].float().cpu().numpy()                  # (N, K, H)
        p = o.prediction_outputs[..., 0].float().cpu().numpy()                # (N, H), the mean
        Q = np.moveaxis(q, 1, 0)            # (K, N, H)
        return _regroup(batch, owners, Q, p, taus)
