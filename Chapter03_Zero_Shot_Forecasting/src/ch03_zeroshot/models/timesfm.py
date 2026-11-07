# Copyright (c) 2026 Shovon Sengupta and Tanujit Chakraborty.
# Companion code for "Hands-On Open-Source Foundation Models for Time Series
# Forecasting" (Packt). Licensed under the MIT License.
"""TimesFM 2.5 and TimesFM 3.0 adapters (Google, ``pip install "timesfm[torch]"``).

Verified against timesfm 3.0.2 source:
* ``timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo)`` then
  ``model.compile(timesfm.ForecastConfig(...))`` and
  ``model.forecast(horizon, inputs: list[np.ndarray]) -> (point (B, H), quantiles (B, H, 10))``.
  Channel 0 of the quantile array is the mean, channels 1..9 are q0.1..q0.9
  and the point forecast is channel 5 (the median).
* Covariates in 2.5 go through ``forecast_with_covariates`` (in-context linear
  "XReg"), which needs ``ForecastConfig(return_backcast=True)``, the ``[xreg]``
  extra (JAX), and dynamic covariates of length T + H (i.e. known-future).
  It returns ``(points, quantiles)`` lists per input.
* ``timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device=...)``
  and ``predict_batch(contexts, horizon, past_only_covariates, past_future_covariates,
  return_quantiles=True)`` yield ``ForecastOutput(forecast, quantiles)`` with
  quantiles shaped (H, 9) for 1-D inputs or (V, H, 9) for (V, T) inputs.
  ``past_future_covariates`` must have length T + H. All contexts in one call
  must have the same number of variates. Max context 15,360.
* TimesFM 3.0 WEIGHTS are under ``timesfm-non-commercial-license-v1.0``
  (stated in the package metadata); the code is Apache-2.0.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastOutput, ZeroShotForecaster, select_quantiles

_TFM_TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _jax_on_cpu():
    """Run JAX on the CPU. Only TimesFM-2.5's covariate step (XReg, a small in-context least-squares fit) uses JAX.

    On a Colab GPU runtime, installing the pinned jax/jaxlib leaves Colab's own JAX CUDA plugin at an older
    version, and the first linear-algebra call then fails with "No FFI handler registered for cusolver_..._ffi".
    The fit is tiny, so the CPU costs nothing; the transformer itself still runs on the GPU through torch.
    """
    import jax

    jax.config.update("jax_platforms", "cpu")


class TimesFM25(ZeroShotForecaster):
    name = "TimesFM-2.5"
    family = "TimesFM"
    checkpoint = "google/timesfm-2.5-200m-pytorch"
    license = "Apache-2.0"
    output = "quantiles"
    supports_future_covariates = True     # via XReg (linear, in-context)
    max_context = 16384
    CONTEXT_LIMIT = 16384                 # the model's own limit on context + horizon

    def _load(self):
        _jax_on_cpu()
        import timesfm

        self._timesfm = timesfm
        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint)
        self._compiled_for = None

    def _compile(self, max_context, horizon, backcast):
        key = (max_context, horizon, backcast)
        if self._compiled_for == key:
            return
        # max_context / max_horizon are rounded up to TimesFM's patch sizes (32 in, 128 out).
        mc = int(np.ceil(max_context / 32) * 32)
        mh = int(np.ceil(horizon / 128) * 128)
        self._model.compile(self._timesfm.ForecastConfig(
            max_context=mc, max_horizon=mh, normalize_inputs=True,
            use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=self.kwargs.get("infer_is_positive", True),
            fix_quantile_crossing=True, return_backcast=backcast))
        self._compiled_for = key

    def _predict(self, batch, horizon, taus):
        # TimesFM-2.5 needs context + horizon (rounded up to 128) <= 16384, so long series lose a few steps.
        limit = min(self.max_context or self.CONTEXT_LIMIT, self.CONTEXT_LIMIT - int(np.ceil(horizon / 128) * 128))
        if any(np.asarray(s.target).shape[-1] > limit for s in batch):
            saved, self.max_context = self.max_context, limit
            try:
                batch = [self._truncate(s) for s in batch]
            finally:
                self.max_context = saved
        flat, owners = [], []
        for i, s in enumerate(batch):
            y = np.asarray(s.target, np.float32)
            rows = [y] if y.ndim == 1 else list(y)
            flat += rows
            owners += [i] * len(rows)
        max_ctx = min(max(len(r) for r in flat), limit)
        use_x = all(s.future_covariates is not None for s in batch) and not any(s.is_multivariate for s in batch)
        self._compile(max_ctx, horizon, backcast=use_x)
        if use_x:
            names = batch[0].future_cov_names
            dyn = {n: [list(s.future_covariates[k][: len(s.target) + horizon]) for s in batch]
                   for k, n in enumerate(names)}
            points, quants = self._model.forecast_with_covariates(
                inputs=[list(r) for r in flat], dynamic_numerical_covariates=dyn,
                xreg_mode=self.kwargs.get("xreg_mode", "xreg + timesfm"))
            point = np.stack([np.asarray(p)[:horizon] for p in points])
            quant = np.stack([np.asarray(q)[:horizon] for q in quants])
        else:
            point, quant = self._model.forecast(horizon=horizon, inputs=flat)
            point, quant = np.asarray(point)[:, :horizon], np.asarray(quant)[:, :horizon]
        native = np.moveaxis(quant[..., 1:10], -1, 0)      # (9, N, H) = q0.1..q0.9
        owners = np.array(owners)
        outs = []
        for i, s in enumerate(batch):
            idx = np.where(owners == i)[0]
            Q = select_quantiles(native[:, idx], _TFM_TAUS, taus)
            m = point[idx]
            if not s.is_multivariate:
                Q, m = Q[:, 0], m[0]
            outs.append(ForecastOutput(s.item_id, mean=m, quantiles=Q, taus=taus,
                                       meta={"covariates_used": use_x}))
        return outs


class TimesFM3(ZeroShotForecaster):
    name = "TimesFM-3"
    family = "TimesFM"
    checkpoint = "google/timesfm-3.0-pytorch"
    license = "timesfm-non-commercial-license-v1.0 (weights); Apache-2.0 (code)"
    commercial_ok = False
    output = "quantiles"
    supports_multivariate = True
    supports_past_covariates = True
    supports_future_covariates = True
    max_context = 15360

    def _load(self):
        import timesfm

        self._model = timesfm.TimesFM3Forecaster.from_pretrained(
            self.checkpoint, device=self.device, per_core_batch_size=self.kwargs.get("batch_size", 32))

    def _predict(self, batch, horizon, taus):
        outs = []
        # one call per number-of-variates group (a TimesFM 3 requirement)
        groups = {}
        for i, s in enumerate(batch):
            groups.setdefault(np.asarray(s.target).shape[0] if s.is_multivariate else 1, []).append(i)
        results = [None] * len(batch)
        for _, idx in groups.items():
            ctx = [np.asarray(batch[i].target, np.float32) for i in idx]
            po = [batch[i].past_covariates for i in idx]
            pf = [None if batch[i].future_covariates is None
                  else batch[i].future_covariates[:, : np.asarray(batch[i].target).shape[-1] + horizon]
                  for i in idx]
            res = self._model.predict_batch(
                contexts=ctx, horizon=horizon,
                past_only_covariates=po if any(p is not None for p in po) else None,
                past_future_covariates=pf if any(p is not None for p in pf) else None,
                return_quantiles=True)
            for i, r in zip(idx, res):
                results[i] = r
        for s, r in zip(batch, results):
            q = np.asarray(r.quantiles, float)          # (H, 9) or (V, H, 9)
            Q = select_quantiles(np.moveaxis(q, -1, 0), _TFM_TAUS, taus)
            outs.append(ForecastOutput(s.item_id, mean=np.asarray(r.forecast, float), quantiles=Q, taus=taus))
        return outs
