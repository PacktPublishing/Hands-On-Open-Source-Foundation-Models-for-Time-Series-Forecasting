# Copyright (c) 2026 Shovon Sengupta and Tanujit Chakraborty.
# Companion code for "Hands-On Open-Source Foundation Models for Time Series
# Forecasting" (Packt). Licensed under the MIT License.
"""Moirai 1.1, Moirai-MoE and Moirai 2.0 adapters (Salesforce ``uni2ts``).

Environment ``moirai`` (see ``requirements/env-moirai.txt``): uni2ts pins
numpy 1.26 and an old gluonts, so it cannot share the core environment.

Verified against uni2ts 2.0.0 source:

* All three wrappers expose ``create_predictor(batch_size, device)``, which
  returns a GluonTS ``PyTorchPredictor``. We use that route (the one in the
  uni2ts README) because it handles left padding, observed masks and
  known-future covariates (``feat_dynamic_real``) correctly.
* ``Moirai2Forecast.predict(...)`` is *not* used: it cuts ``feat_dynamic_real``
  to the context length, so known-future covariates fail, and its multivariate
  path crashes when a series is shorter than the context
  (``uni2ts/model/moirai2/forecast.py`` lines 448-511).
* Moirai 1.1 / MoE return 100 sample paths by default (a mixture
  distribution); Moirai 2.0 returns the 9 deciles directly.
* Moirai does not read timestamps or the frequency, so every series is given a
  dummy hourly start; only the order of values matters.
* Licence: all Moirai checkpoints are CC-BY-NC-4.0 (non-commercial).
"""
from __future__ import annotations

import numpy as np

from .base import CannotForecast, ForecastOutput, ZeroShotForecaster, quantiles_from_samples, select_quantiles


class _MoiraiBase(ZeroShotForecaster):
    family = "Moirai"
    license = "CC-BY-NC-4.0"
    commercial_ok = False
    supports_multivariate = True
    supports_past_covariates = True
    supports_future_covariates = True
    max_context = 2048          # harness default (configurable); uni2ts accepts any length
    batch_size = 32

    def __init__(self, checkpoint=None, device="cpu", num_samples: int = 100, module=None,
                 max_context: int | None = None, **kw):
        super().__init__(checkpoint=checkpoint, device=device, **kw)
        self.num_samples = num_samples
        self._module = module        # tests inject a tiny randomly-initialised module here
        if max_context is not None:
            self.max_context = max_context

    # subclasses return (ForecastClass, ModuleClass, extra forecast kwargs)
    def _classes(self):  # pragma: no cover
        raise NotImplementedError

    def _load(self):
        _, module_cls, _ = self._classes()
        module = self._module if self._module is not None else module_cls.from_pretrained(self.checkpoint)
        self._model = module.eval()

    def _wrapper(self, horizon, ctx_len, target_dim, n_fdr, n_pfdr, num_samples=None):
        forecast_cls, _, extra = self._classes()
        if num_samples is not None and "num_samples" in extra:
            extra = {**extra, "num_samples": num_samples}
        return forecast_cls(module=self._model, prediction_length=horizon, context_length=ctx_len,
                            target_dim=target_dim, feat_dynamic_real_dim=n_fdr,
                            past_feat_dynamic_real_dim=n_pfdr, **extra).eval()

    def _forecasts(self, entries, horizon, ctx_len, V, kf, kp):
        """GluonTS forecasts for ``entries``, one wrapper and one predictor call."""
        from gluonts.dataset.common import ListDataset

        ds = ListDataset(entries, freq="h", one_dim_target=(V == 1))
        wrapper = self._wrapper(horizon, ctx_len, V, kf, kp)
        return list(wrapper.create_predictor(batch_size=self.batch_size, device=self.device).predict(ds))

    def _predict(self, batch, horizon, taus):
        import pandas as pd

        # The wrapper's dimensions are fixed per call, so group compatible series.
        groups: dict[tuple, list[int]] = {}
        for i, s in enumerate(batch):
            V = 1 if not s.is_multivariate else np.asarray(s.target).shape[0]
            kf = 0 if s.future_covariates is None else len(s.future_covariates)
            kp = 0 if s.past_covariates is None else len(s.past_covariates)
            groups.setdefault((V, kf, kp), []).append(i)

        outs: list[ForecastOutput | None] = [None] * len(batch)
        start = pd.Period("2000-01-01 00:00", freq="h")
        for (V, kf, kp), idx in groups.items():
            entries = []
            for i in idx:
                s = batch[i]
                e = {"start": start, "target": np.asarray(s.target, np.float32)}
                if kf:
                    e["feat_dynamic_real"] = np.asarray(s.future_covariates, np.float32)[:, : np.asarray(
                        s.target).shape[-1] + horizon]
                if kp:
                    e["past_feat_dynamic_real"] = np.asarray(s.past_covariates, np.float32)
                entries.append(e)
            ctx_len = max(np.asarray(batch[i].target).shape[-1] for i in idx)
            fcs = self._forecasts(entries, horizon, ctx_len, V, kf, kp)
            for i, fc in zip(idx, fcs):
                outs[i] = self._to_output(batch[i], fc, taus)
        return outs

    def _to_output(self, s, fc, taus):
        raise NotImplementedError  # pragma: no cover


class _SampleMoirai(_MoiraiBase):
    output = "samples"

    def _to_output(self, s, fc, taus):
        S = np.asarray(fc.samples, float)               # (S, H) or (S, H, V)
        if S.ndim == 3:
            S = np.moveaxis(S, -1, 1)                   # (S, V, H)
        Q = quantiles_from_samples(S, taus)
        return ForecastOutput(s.item_id, mean=np.median(S, axis=0), quantiles=Q, taus=taus, samples=S)


class Moirai11(_SampleMoirai):
    name = "Moirai-1.1"
    checkpoint = "Salesforce/moirai-1.1-R-small"
    patch_size = 32

    def _classes(self):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        # patch_size=32 as in the GIFT-Eval Moirai notebook. "auto" failed with a size mismatch
        # (64 vs 70) for a 512-step context and H=48 on the real checkpoint (uni2ts 2.0.0, 16 Oct 2026).
        return MoiraiForecast, MoiraiModule, {"patch_size": self.patch_size, "num_samples": self.num_samples}


class _Samples:
    """The part of a GluonTS SampleForecast that ``_to_output`` reads."""

    def __init__(self, samples):
        self.samples = samples


class MoiraiMoE(_SampleMoirai):
    """Moirai-MoE, a decoder-only model that forecasts patch by patch.

    After the first forecast patch, uni2ts 2.0.0 re-runs the decoder on ``num_samples`` copies of the whole
    input (``MoiraiMoEForecast.forward``: ``target.unsqueeze(0).repeat(num_samples, ...)``), and every
    target channel and covariate is a separate row of patches in that input. Memory therefore grows with
    series x samples x tokens^2, where tokens = variates x (context + horizon) / 16. Two guards follow:

    * a call is split (fewer series per call, then fewer samples per call) so that series x samples x
      tokens^2 stays below ``budget``; all ``num_samples`` samples are still drawn, so the forecast
      distribution is the same. The univariate runs of the ``cpu`` profile (3 series, 130 tokens) are
      below the budget and run in one call, as before;
    * on a CPU, inputs of more than ``max_tokens_cpu`` tokens are skipped with a reason. E3's bike_oracle
      (one target and nine covariates, 1,300 tokens at a 2,048-step context) passed 13.6 GB and was
      killed on a 15 GB, 4-core machine (16 Oct 2026); the cost grows with the square of the tokens.
    """

    name = "Moirai-MoE"
    checkpoint = "Salesforce/moirai-moe-1.0-R-small"
    patch_size = 16
    budget = 3e7                # series x samples x tokens^2 per decoder call
    max_tokens_cpu = 640        # five variates at a 2,048-step context

    def _tokens(self, ctx_len, horizon, n_variates):
        return n_variates * (-(-ctx_len // self.patch_size) + -(-horizon // self.patch_size))

    def _forecasts(self, entries, horizon, ctx_len, V, kf, kp):
        tokens = self._tokens(ctx_len, horizon, V + kf + kp)
        if self.device == "cpu" and tokens > self.max_tokens_cpu:
            # Read a shorter context so the input fits the CPU token cap (most recent steps kept; the
            # predictor slices the last ctx_len steps of each entry). Skip only if fewer than 4 patches fit.
            patches = self.max_tokens_cpu // (V + kf + kp) - -(-horizon // self.patch_size)
            if patches >= 4:
                short = patches * self.patch_size
                self.notes = getattr(self, "notes", set()) | {
                    f"CPU: {V + kf + kp} variates, context shortened from {ctx_len} to {short} steps "
                    f"(token cap {self.max_tokens_cpu})"}
                ctx_len, tokens = short, self._tokens(short, horizon, V + kf + kp)
        if self.device == "cpu" and tokens > self.max_tokens_cpu:
            raise CannotForecast(
                f"Moirai-MoE on a CPU: {V + kf + kp} variates (target channels and covariates) at a {ctx_len}-step "
                f"context are {tokens} tokens, above {self.max_tokens_cpu}; its {self.num_samples}-sample "
                f"autoregressive decoding needs memory in proportion to samples x tokens^2 (run it on a GPU)")
        one_call = super()._forecasts
        per_series = self.num_samples * tokens ** 2
        if len(entries) * per_series <= self.budget:
            return one_call(entries, horizon, ctx_len, V, kf, kp)
        if per_series <= self.budget:               # fewer series per call, every sample in one call
            n = max(1, int(self.budget // per_series))
            return [fc for k in range(0, len(entries), n)
                    for fc in one_call(entries[k: k + n], horizon, ctx_len, V, kf, kp)]
        from gluonts.dataset.common import ListDataset

        chunk = max(1, int(self.budget // tokens ** 2))  # one series per call, the samples in chunks
        out = []
        for e in entries:
            ds = ListDataset([e], freq="h", one_dim_target=(V == 1))
            draws, left = [], self.num_samples
            while left > 0:
                k = min(chunk, left)
                wrapper = self._wrapper(horizon, ctx_len, V, kf, kp, num_samples=k)
                fc = next(iter(wrapper.create_predictor(batch_size=1, device=self.device).predict(ds)))
                draws.append(np.asarray(fc.samples))
                left -= k
            out.append(_Samples(np.concatenate(draws, axis=0)))
        return out

    def _classes(self):
        from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

        # README: Moirai-MoE uses a fixed patch size of 16 (it has no "auto" search)
        return MoiraiMoEForecast, MoiraiMoEModule, {"patch_size": self.patch_size, "num_samples": self.num_samples}


class Moirai2(_MoiraiBase):
    """Moirai 2.0. Multivariate targets are forecast channel by channel.

    uni2ts 2.0.0 marks ``Moirai2Forecast.predict`` as "only support univariate
    forecast now". A direct ``forward`` call with ``target_dim=7`` on the real
    checkpoint did return ``(1, 9, 48, 7)`` (16 Oct 2026), but this adapter stays on the
    documented path: it splits the channels and keeps any covariates attached to
    every channel, so Moirai 2.0 is not part of the joint-vs-CI comparison (E4).
    """

    name = "Moirai-2.0"
    checkpoint = "Salesforce/moirai-2.0-R-small"
    output = "quantiles"
    supports_multivariate = False
    # The module keeps Moirai's variate ids and attention bias, so uni2ts would accept extra
    # covariate rows, but the paper describes Moirai 2.0 as univariate ("Currently, Moirai 2.0
    # is designed for univariate time-series input", Liu et al. 2025, Section 3.1) and it was
    # pre-trained that way. We therefore pass no covariates.
    supports_past_covariates = False
    supports_future_covariates = False
    max_context = 1680          # the value used in the uni2ts README example

    def _predict(self, batch, horizon, taus):
        from .base import SeriesInput

        flat, owners = [], []
        for i, s in enumerate(batch):
            rows = [s.target] if not s.is_multivariate else list(np.asarray(s.target))
            for r in rows:
                flat.append(SeriesInput(s.item_id, np.asarray(r), s.freq, s.start, s.past_covariates,
                                        s.future_covariates, s.past_cov_names, s.future_cov_names, s.static))
                owners.append(i)
        res = super()._predict(flat, horizon, taus)
        owners = np.asarray(owners)
        outs = []
        for i, s in enumerate(batch):
            parts = [res[j] for j in np.where(owners == i)[0]]
            if not s.is_multivariate:
                outs.append(parts[0])
                continue
            outs.append(ForecastOutput(s.item_id, mean=np.stack([p.mean for p in parts]),
                                       quantiles=np.stack([p.quantiles for p in parts], axis=1), taus=taus))
        return outs

    def _classes(self):
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        return Moirai2Forecast, Moirai2Module, {}

    def _to_output(self, s, fc, taus):
        levels = [float(k) for k in fc.forecast_keys]
        arr = np.asarray(fc.forecast_array, float)      # (9, H) or (9, H, V)
        if arr.ndim == 3:
            arr = np.moveaxis(arr, -1, 1)               # (9, V, H)
        Q = select_quantiles(arr, levels, taus)
        med = select_quantiles(arr, levels, (0.5,))[0]
        return ForecastOutput(s.item_id, mean=med, quantiles=Q, taus=taus)
