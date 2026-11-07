# Copyright (c) 2026 Shovon Sengupta and Tanujit Chakraborty.
# Companion code for "Hands-On Open-Source Foundation Models for Time Series
# Forecasting" (Packt). Licensed under the MIT License.
"""Lag-Llama adapter (env ``lagllama``: Python 3.10, gluonts 0.14.4).

Lag-Llama is installed from its GitHub repository (it is not on PyPI). The
flow below follows the official zero-shot Colab demo and was checked against
the repository at commit df7531a:

* download ``lag-llama.ckpt`` from ``time-series-foundation-models/Lag-Llama``;
* read ``ckpt["hyper_parameters"]["model_kwargs"]`` for the architecture;
* build ``LagLlamaEstimator(ckpt_path=..., prediction_length=H, context_length=C,
  ..., rope_scaling={"type": "linear", "factor": max(1, (C + H) / C_train)})``;
* ``create_predictor(create_transformation(), create_lightning_module())`` gives
  a GluonTS predictor whose forecasts hold ``num_parallel_samples`` paths.

The model was trained with a context of 32 (README "best practices"), which is
the default here; longer contexts use linear RoPE scaling.

PyTorch >= 2.6 refuses to unpickle the checkpoint (it stores gluonts objects),
so the adapter sets ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` before importing
Lightning and calls ``torch.load(..., weights_only=False)``. Only use
checkpoints you trust with this setting.

Pip-installing the repository also installs top-level packages called
``data``, ``utils`` and ``gluon_utils``; keep that environment separate.
"""
from __future__ import annotations

import os

import numpy as np

from .base import ForecastOutput, ZeroShotForecaster, quantiles_from_samples


class LagLlama(ZeroShotForecaster):
    name = "Lag-Llama"
    family = "Lag-Llama"
    checkpoint = "time-series-foundation-models/Lag-Llama"
    license = "Apache-2.0"
    output = "samples"

    def __init__(self, checkpoint=None, device="cpu", context_length: int = 32, num_samples: int = 100,
                 ckpt_path: str | None = None, batch_size: int = 64, **kw):
        super().__init__(checkpoint=checkpoint, device=device, **kw)
        self.context_length = context_length
        self.num_samples = num_samples
        self.ckpt_path = ckpt_path
        self.batch_size = batch_size

    def _load(self):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        import torch

        if self.ckpt_path is None:
            from huggingface_hub import hf_hub_download

            self.ckpt_path = hf_hub_download(self.checkpoint, "lag-llama.ckpt")
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self._args = ckpt["hyper_parameters"]["model_kwargs"]
        self._model = True        # the estimator is rebuilt per horizon in _predict

    def _predictor(self, horizon):
        import torch
        from lag_llama.gluon.estimator import LagLlamaEstimator

        a = self._args
        C = self.context_length
        est = LagLlamaEstimator(
            ckpt_path=self.ckpt_path, prediction_length=horizon, context_length=C,
            input_size=a["input_size"], n_layer=a["n_layer"], n_embd_per_head=a["n_embd_per_head"],
            n_head=a["n_head"], scaling=a["scaling"], time_feat=a["time_feat"],
            rope_scaling={"type": "linear", "factor": max(1.0, (C + horizon) / a["context_length"])},
            batch_size=self.batch_size, num_parallel_samples=self.num_samples,
            device=torch.device(self.device))
        module = est.create_lightning_module()
        self._net = module        # kept so the runner can count the weights
        return est.create_predictor(est.create_transformation(), module)

    def _predict(self, batch, horizon, taus):
        import pandas as pd
        from gluonts.dataset.common import ListDataset

        # The checkpoint has time_feat=True: besides its 84 lags, every token carries six calendar
        # features (second of minute up to day of year) computed from the timestamps, so each
        # series keeps its real start date and frequency (pandas Period uses "M" for "MS").
        freq = {"MS": "M"}.get(batch[0].freq, batch[0].freq)
        rows, starts, owners = [], [], []
        for i, s in enumerate(batch):
            y = np.asarray(s.target, float)
            for r in ([y] if y.ndim == 1 else list(y)):
                rows.append(r)
                starts.append(pd.Period(pd.Timestamp(s.start), freq=freq))
                owners.append(i)
        owners = np.asarray(owners)
        ds = ListDataset([{"start": st, "target": r.astype(np.float32)} for st, r in zip(starts, rows)], freq=freq)
        samples = [np.asarray(f.samples, float) for f in self._predictor(horizon).predict(ds)]   # (S, H)
        outs = []
        for i, s in enumerate(batch):
            parts = [samples[j] for j in np.where(owners == i)[0]]
            smp = parts[0] if not s.is_multivariate else np.stack(parts, axis=1)               # (S, V, H)
            outs.append(ForecastOutput(s.item_id, mean=np.median(smp, axis=0),
                                       quantiles=quantiles_from_samples(smp, taus), taus=taus, samples=smp))
        return outs
