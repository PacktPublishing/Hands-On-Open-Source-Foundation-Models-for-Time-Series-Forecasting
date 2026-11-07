# Copyright (c) 2026 Shovon Sengupta and Tanujit Chakraborty.
# Companion code for "Hands-On Open-Source Foundation Models for Time Series
# Forecasting" (Packt). Licensed under the MIT License.
"""Generative "LLM-style" forecasters loaded with ``trust_remote_code``.

All of them are decoder-only models whose modelling code lives in the Hugging
Face repository, so the ``transformers`` version matters:

* Sundial (``thuml/sundial-base-128m``), Timer (``thuml/timer-base-84m``) and
  Time-MoE (``Maple728/TimeMoE-50M``): ``transformers==4.40.1`` (env ``thuml``).
  Timer's remote code fails on transformers 4.57 (``DynamicCache`` has no
  ``seen_tokens``), so the pin is real.
* Timer-S1 (``bytedance-research/Timer-S1``): ``transformers~=4.57.1``
  (env ``timer-s1``), 8.3B parameters, needs a large GPU.
* YingLong (``qcw2333/YingLong_*``): no pin in its README; bf16 on a GPU.

Call conventions (checked against the model code or official notebooks):

* Sundial ``generate(x (B, L), max_new_tokens=H, num_samples=S, revin=True)``
  returns ``(B, S, H)`` sample paths in the original scale.
* Timer ``generate(x (B, L), max_new_tokens=H)`` returns ``(B, H)``; the input
  is cut from the left to a multiple of 96 and must have at least 96 points.
* Time-MoE ``generate(x_norm (B, L), max_new_tokens=H)`` returns ``(B, L + H)``
  and does **not** normalise internally, so the adapter standardises each
  series and inverts afterwards (README). Context + horizon should stay <= 4096.
* YingLong ``generate(x (B, L), future_token=H)`` returns ``(B, H, 99)``
  quantile channels at levels 0.01 ... 0.99 (index 49 = median, notebook).
  We use the notebook's "DCoT" recipe: ask for 4096 future tokens and keep the
  first H.
* Timer-S1 ``generate(x (B, L), max_new_tokens=H, revin=True)`` under bf16
  autocast with ``use_cache=False`` returns ``(B, 9, H)`` (GIFT-Eval notebook).

All five take a dense 2-D batch, so missing values are linearly interpolated
and series are grouped by length before each call.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastOutput, ZeroShotForecaster, quantiles_from_samples, select_quantiles


def dense(y) -> np.ndarray:
    """Linear interpolation of NaNs (edges filled with the nearest value)."""
    y = np.asarray(y, float)
    if not np.isnan(y).any():
        return y
    idx = np.arange(len(y))
    ok = ~np.isnan(y)
    if not ok.any():
        return np.zeros_like(y)
    return np.interp(idx, idx[ok], y[ok])


class _HFGenerative(ZeroShotForecaster):
    """Shared plumbing: univariate rows, grouped by length, through ``generate``."""

    family = "decoder-only"
    license = "Apache-2.0"
    torch_dtype = None

    def __init__(self, checkpoint=None, device="cpu", model=None, batch_size: int = 32, **kw):
        super().__init__(checkpoint=checkpoint, device=device, **kw)
        self._injected = model
        self.batch_size = batch_size

    def _load(self):
        if self._injected is not None:
            self._model = self._injected
            return
        import torch
        from transformers import AutoModelForCausalLM

        kw = {"trust_remote_code": True}
        if self.torch_dtype is not None:
            kw["torch_dtype"] = getattr(torch, self.torch_dtype)
        self._model = AutoModelForCausalLM.from_pretrained(self.checkpoint, **kw).to(self.device).eval()

    def _rows(self, batch):
        rows, owners = [], []
        for i, s in enumerate(batch):
            y = np.asarray(s.target, float)
            for r in ([y] if y.ndim == 1 else list(y)):
                rows.append(dense(r))
                owners.append(i)
        return rows, np.asarray(owners)

    def _generate(self, x, horizon):  # pragma: no cover - per model
        """x: float32 tensor (B, L). Return a numpy array (B, K, H) of quantiles/samples or (B, H)."""
        raise NotImplementedError

    def _call_rows(self, rows, horizon):
        import torch

        out = [None] * len(rows)
        by_len: dict[int, list[int]] = {}
        for j, r in enumerate(rows):
            by_len.setdefault(len(r), []).append(j)
        for _, idx in by_len.items():
            for k in range(0, len(idx), self.batch_size):
                part = idx[k: k + self.batch_size]
                x = torch.tensor(np.stack([rows[j] for j in part]), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    res = np.asarray(self._generate(x, horizon), float)
                for j, r in zip(part, res):
                    out[j] = r
        return out

    def _assemble(self, batch, owners, per_row, taus):
        """per_row[j] -> (quantiles (K, H), mean (H,), samples (S, H) | None)."""
        outs = []
        for i, s in enumerate(batch):
            parts = [per_row[j] for j in np.where(owners == i)[0]]
            if not s.is_multivariate:
                Q, mean, smp = parts[0]
            else:
                Q = np.stack([p[0] for p in parts], axis=1)
                mean = np.stack([p[1] for p in parts])
                smp = None if parts[0][2] is None else np.stack([p[2] for p in parts], axis=1)
            outs.append(ForecastOutput(s.item_id, mean=mean, quantiles=Q, taus=taus, samples=smp))
        return outs


class Sundial(_HFGenerative):
    name = "Sundial"
    family = "Sundial"
    checkpoint = "thuml/sundial-base-128m"
    output = "samples"
    max_context = 2880        # look-back used in the Sundial README benchmarks

    def __init__(self, checkpoint=None, device="cpu", num_samples: int = 100, **kw):
        super().__init__(checkpoint=checkpoint, device=device, **kw)
        self.num_samples = num_samples

    def _generate(self, x, horizon):
        return self._model.generate(x, max_new_tokens=horizon, num_samples=self.num_samples,
                                    revin=True).float().cpu().numpy()          # (B, S, H)

    def _predict(self, batch, horizon, taus):
        rows, owners = self._rows(batch)
        res = self._call_rows(rows, horizon)
        per_row = [(quantiles_from_samples(r, taus), np.median(r, axis=0), r) for r in res]
        return self._assemble(batch, owners, per_row, taus)


class Timer(_HFGenerative):
    name = "Timer"
    family = "Timer"
    checkpoint = "thuml/timer-base-84m"
    output = "point"
    max_context = 2880
    min_context = 96          # generate() raises for fewer than input_token_len (96) points

    def _generate(self, x, horizon):
        return self._model.generate(x, max_new_tokens=horizon).float().cpu().numpy()   # (B, H)

    def _predict(self, batch, horizon, taus):
        rows, owners = self._rows(batch)
        res = self._call_rows(rows, horizon)
        per_row = [(None, r, None) for r in res]
        outs = self._assemble_point(batch, owners, per_row, taus)
        return outs

    def _assemble_point(self, batch, owners, per_row, taus):
        outs = []
        for i, s in enumerate(batch):
            parts = [per_row[j][1] for j in np.where(owners == i)[0]]
            mean = parts[0] if not s.is_multivariate else np.stack(parts)
            outs.append(ForecastOutput(s.item_id, mean=mean, quantiles=None, taus=taus))
        return outs


class TimeMoE(Timer):
    name = "Time-MoE"
    family = "Time-MoE"
    checkpoint = "Maple728/TimeMoE-50M"
    max_context = 4096        # context + horizon is capped at 4096 inside _predict
    min_context = None

    def __init__(self, checkpoint=None, device="cpu", batch_size: int = 5, **kw):
        # Every time step is a token, so memory grows with series x context^2: five 2,048-step
        # series peaked at 2.8 GB on CPU; 25 in one call passed 12 GB and were killed (3 Nov 2026).
        super().__init__(checkpoint, device, batch_size=batch_size, **kw)

    def _generate(self, x, horizon):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-8)   # README: normalise yourself
        out = self._model.generate((x - mean) / std, max_new_tokens=horizon)
        return (out[:, -horizon:] * std + mean).float().cpu().numpy()

    def _predict(self, batch, horizon, taus):
        rows, owners = self._rows(batch)
        cap = max(1, 4096 - horizon)
        rows = [r[-cap:] for r in rows]
        res = self._call_rows(rows, horizon)
        return self._assemble_point(batch, owners, [(None, r, None) for r in res], taus)


class YingLong(_HFGenerative):
    name = "YingLong"
    family = "YingLong"
    checkpoint = "qcw2333/YingLong_110m"
    license = "CC-BY-4.0"
    output = "quantiles"
    max_context = 2880        # notebook: "support the maximum context length up to 2880"
    torch_dtype = "bfloat16"
    _LEVELS = tuple(np.round(np.arange(1, 100) / 100, 2))

    def _generate(self, x, horizon):
        import torch

        x = x.to(torch.bfloat16)
        out = self._model.generate(x, future_token=max(horizon, 4096))     # DCoT recipe
        return out[:, :horizon, :].float().cpu().numpy().transpose(0, 2, 1)  # (B, 99, H)

    def _predict(self, batch, horizon, taus):
        rows, owners = self._rows(batch)
        res = self._call_rows(rows, horizon)
        per_row = [(select_quantiles(r, self._LEVELS, taus), r[49], None) for r in res]
        return self._assemble(batch, owners, per_row, taus)


class TimerS1(_HFGenerative):
    name = "Timer-S1"
    family = "Timer"
    checkpoint = "bytedance-research/Timer-S1"
    output = "quantiles"
    max_context = 11520       # MAX_CONTEXT_LENGTH in the GIFT-Eval Timer-S1 notebook
    torch_dtype = "bfloat16"

    def _load(self):
        super()._load()
        self._model.config.use_cache = False

    def _generate(self, x, horizon):
        import torch

        with torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu",
                            dtype=torch.bfloat16):
            out = self._model.generate(x, max_new_tokens=horizon, revin=True)
        return out.float().cpu().numpy()                                      # (B, 9, H)

    def _predict(self, batch, horizon, taus):
        rows, owners = self._rows(batch)
        res = self._call_rows(rows, horizon)
        levels = tuple(getattr(self._model.config, "quantiles", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))
        per_row = [(select_quantiles(r, levels, taus), select_quantiles(r, levels, (0.5,))[0], None) for r in res]
        return self._assemble(batch, owners, per_row, taus)
