# Copyright (c) 2026 Shovon Sengupta and Tanujit Chakraborty.
# Companion code for "Hands-On Open-Source Foundation Models for Time Series
# Forecasting" (Packt). Licensed under the MIT License.
"""Kairos adapter (env ``kairos``; install from github.com/foundation-model-research/Kairos).

Checked against the repository source at commit 16eb4b9:

* The import name is ``tsfm``: ``from tsfm.model.kairos import AutoModel`` registers
  the Kairos classes with transformers and re-exports its ``AutoModel``.
* ``model(past_target=(B, L), past_is_pad=(B, L), prediction_length=H,
  generation=True, infer_is_positive=..., force_flip_invariance=...)`` returns
  ``prediction_outputs`` shaped ``(B, 9, H)`` at ``config.quantiles``
  (0.1 ... 0.9 by default), see ``tsfm/model/kairos/modeling_kairos.py:398-573``.
* The README passes ``preserve_positivity=True, average_with_flipped_input=True``.
  Those keyword names do not exist and are silently swallowed by ``**kwargs``;
  the real names are ``infer_is_positive`` and ``force_flip_invariance``, which
  is what this adapter uses.
* NaNs are treated as missing (masked). Even so, a shorter series left-padded inside
  a batch got a very different forecast from the same series alone (checked on the real
  Kairos_50m checkpoint), so the adapter forecasts each context length separately.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastOutput, ZeroShotForecaster, select_quantiles

_DEFAULT_TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class Kairos(ZeroShotForecaster):
    name = "Kairos"
    family = "Kairos"
    checkpoint = "mldi-lab/Kairos_50m"
    license = "Apache-2.0"
    output = "quantiles"
    max_context = 2048          # KairosConfig.context_length default; the model truncates to it
    pad_free = True             # padded + masked series moved by 1.6 sd (check_models, 16 Oct 2026)

    def __init__(self, checkpoint=None, device="cpu", model=None, infer_is_positive: bool = True,
                 force_flip_invariance: bool = True, batch_size: int = 64, **kw):
        super().__init__(checkpoint=checkpoint, device=device, **kw)
        self._injected = model
        self.infer_is_positive = infer_is_positive
        self.force_flip_invariance = force_flip_invariance
        self.batch_size = batch_size

    def _load(self):
        if self._injected is not None:
            model = self._injected
        else:
            from tsfm.model.kairos import AutoModel

            model = AutoModel.from_pretrained(self.checkpoint, trust_remote_code=True)
        self._model = model.to(self.device).eval()
        self._levels = tuple(getattr(self._model.config, "quantiles", _DEFAULT_TAUS))

    def _predict(self, batch, horizon, taus):
        import torch

        rows, owners = [], []
        for i, s in enumerate(batch):
            y = np.asarray(s.target, float)
            for r in ([y] if y.ndim == 1 else list(y)):
                rows.append(r)
                owners.append(i)
        owners = np.asarray(owners)
        res = []
        for k in range(0, len(rows), self.batch_size):
            part = rows[k: k + self.batch_size]
            T = max(len(r) for r in part)
            x = np.full((len(part), T), np.nan, np.float32)
            pad = np.ones((len(part), T), bool)
            for j, r in enumerate(part):
                x[j, T - len(r):] = r
                pad[j, T - len(r):] = False
            with torch.no_grad():
                out = self._model(past_target=torch.tensor(x, device=self.device),
                                  past_is_pad=torch.tensor(pad, device=self.device),
                                  prediction_length=horizon, generation=True,
                                  infer_is_positive=self.infer_is_positive,
                                  force_flip_invariance=self.force_flip_invariance)
            res += list(out["prediction_outputs"].float().cpu().numpy())          # (9, H) each
        outs = []
        for i, s in enumerate(batch):
            parts = [res[j] for j in np.where(owners == i)[0]]
            native = parts[0] if not s.is_multivariate else np.stack(parts, axis=1)
            Q = select_quantiles(native, self._levels, taus)
            med = select_quantiles(native, self._levels, (0.5,))[0]
            outs.append(ForecastOutput(s.item_id, mean=med, quantiles=Q, taus=taus))
        return outs
