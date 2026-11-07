# Copyright (c) 2026 Shovon Sengupta and Tanujit Chakraborty.
# Companion code for "Hands-On Open-Source Foundation Models for Time Series
# Forecasting" (Packt). Licensed under the MIT License.
"""The unified zero-shot forecaster interface (Chapter 3, Listing 3.4).

Every model adapter turns one library's native API into the same contract:

    load(device)                     -> None
    predict(batch, horizon, taus)    -> list[ForecastOutput]

so the benchmark code never needs to know which model it is calling. Adapters
must not see the evaluation window: the runner only passes the context.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class CannotForecast(ValueError):
    """The model cannot forecast this input (a skip with a reason, not a failure); the runner logs it."""


@dataclass
class SeriesInput:
    """One forecasting problem as seen at the forecast origin.

    target            : (T,) univariate or (V, T) multivariate context
    past_covariates   : (K_p, T) observed only up to the origin
    future_covariates : (K_f, T + H) known for the context AND the horizon
    """

    item_id: str
    target: np.ndarray
    freq: str
    start: object = None  # pandas Timestamp of target[..., 0]
    past_covariates: np.ndarray | None = None
    future_covariates: np.ndarray | None = None
    past_cov_names: list | None = None
    future_cov_names: list | None = None
    static: dict | None = None

    @property
    def is_multivariate(self) -> bool:
        return np.asarray(self.target).ndim == 2


@dataclass
class ForecastOutput:
    item_id: str
    mean: np.ndarray                          # (H,) or (V, H); median when no mean is produced
    quantiles: np.ndarray | None = None       # (K, H) or (K, V, H) at levels `taus`
    taus: tuple = QUANTILES
    samples: np.ndarray | None = None         # (S, H) or (S, V, H)
    meta: dict = field(default_factory=dict)


class _PeakRSS:
    """Peak increase of the process's resident memory during a block (CPU), sampled every 5 ms.

    Needs ``psutil``; without it ``peak_increase`` is None. On CUDA the adapter reports
    ``torch.cuda.max_memory_allocated()`` instead.
    """

    def __enter__(self):
        import threading

        self.peak_increase = None
        try:
            import psutil
        except ImportError:
            self._thread = None
            return self
        proc = psutil.Process()
        self._start = self._peak = proc.memory_info().rss
        self._stop = threading.Event()

        def watch():
            while not self._stop.is_set():
                self._peak = max(self._peak, proc.memory_info().rss)
                self._stop.wait(0.005)

        self._thread = threading.Thread(target=watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self.peak_increase = max(0, self._peak - self._start)
        return False


class ZeroShotForecaster:
    """Base class. Subclasses set the class attributes and implement ``_load`` / ``_predict``."""

    name: str = "base"
    family: str = ""
    checkpoint: str = ""
    license: str = ""
    commercial_ok: bool = True
    output: str = "quantiles"       # "quantiles" | "samples" | "point"
    supports_multivariate: bool = False
    supports_past_covariates: bool = False
    supports_future_covariates: bool = False
    max_context: int | None = None
    min_context: int | None = None
    max_horizon: int | None = None  # longest horizon the model forecasts natively (None = no limit)
    # True for models whose forecast changes when a shorter series is left-padded and
    # masked inside a batch (found by scripts/check_models.py). The base class then
    # calls ``_predict`` once per context length, so these models never see padding.
    pad_free: bool = False

    def __init__(self, checkpoint: str | None = None, device: str = "cpu", **kwargs):
        if checkpoint:
            self.checkpoint = checkpoint
        self.device = device
        self.kwargs = kwargs
        self._model = None

    # -- public API ----------------------------------------------------------
    def load(self):
        t0 = time.perf_counter()
        self._load()
        self.load_seconds = time.perf_counter() - t0
        return self

    def predict(self, batch: list[SeriesInput], horizon: int, taus=QUANTILES) -> list[ForecastOutput]:
        if self._model is None:
            self.load()
        batch = [self._truncate(s) for s in batch]
        cuda = self.device.startswith("cuda")
        if cuda:
            import torch

            torch.cuda.reset_peak_memory_stats()
        with _PeakRSS() as mem:
            t0 = time.perf_counter()
            outs = self._predict_grouped(batch, horizon, tuple(taus))
            elapsed = time.perf_counter() - t0
        if cuda:
            peak = torch.cuda.max_memory_allocated()
        else:
            peak = mem.peak_increase
        for o in outs:
            o.meta.update(latency_ms_per_series=1000 * elapsed / max(len(batch), 1),
                          peak_mem_mb=None if peak is None else peak / 2**20, model=self.name,
                          checkpoint=self.checkpoint)
            if o.quantiles is not None:
                o.quantiles = np.maximum.accumulate(np.asarray(o.quantiles, float), axis=0)
        return outs

    # -- helpers -------------------------------------------------------------
    def _predict_grouped(self, batch, horizon, taus):
        if not self.pad_free:
            return self._predict(batch, horizon, taus)
        by_len: dict[int, list[int]] = {}
        for i, s in enumerate(batch):
            by_len.setdefault(np.asarray(s.target).shape[-1], []).append(i)
        outs = [None] * len(batch)
        for idx in by_len.values():
            for i, o in zip(idx, self._predict([batch[i] for i in idx], horizon, taus)):
                outs[i] = o
        return outs

    def _truncate(self, s: SeriesInput) -> SeriesInput:
        if self.max_context is None:
            return s
        T = np.asarray(s.target).shape[-1]
        if T <= self.max_context:
            return s
        cut = T - self.max_context
        fut = s.future_covariates[..., cut:] if s.future_covariates is not None else None
        past = s.past_covariates[..., cut:] if s.past_covariates is not None else None
        start = s.start
        if start is not None:
            import pandas as pd

            start = pd.Timestamp(start) + cut * pd.tseries.frequencies.to_offset(s.freq)
        return SeriesInput(s.item_id, np.asarray(s.target)[..., cut:], s.freq, start, past, fut,
                           s.past_cov_names, s.future_cov_names, s.static)

    def set_season_length(self, m: int):
        """Let one loaded model serve datasets with different seasonal periods."""
        if hasattr(self, "season_length"):
            self.season_length = m

    def _load(self):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _predict(self, batch, horizon, taus):  # pragma: no cover
        raise NotImplementedError

    def capabilities(self) -> dict:
        return dict(name=self.name, checkpoint=self.checkpoint, license=self.license,
                    commercial_ok=self.commercial_ok, output=self.output,
                    multivariate=self.supports_multivariate,
                    past_covariates=self.supports_past_covariates,
                    future_covariates=self.supports_future_covariates,
                    max_context=self.max_context, min_context=self.min_context, max_horizon=self.max_horizon)


# --------------------------------------------------------------------------- #
# output conversion helpers shared by adapters
# --------------------------------------------------------------------------- #
def select_quantiles(native_q: np.ndarray, native_taus, taus) -> np.ndarray:
    """Pick (or linearly interpolate in tau) the requested levels from native quantiles.

    native_q has the quantile axis first: (K_native, ...).
    """
    native_taus = np.asarray(native_taus, float)
    native_q = np.asarray(native_q, float)
    out = []
    for t in taus:
        hit = np.where(np.isclose(native_taus, t))[0]
        if len(hit):
            out.append(native_q[hit[0]])
            continue
        j = np.searchsorted(native_taus, t)
        j = int(np.clip(j, 1, len(native_taus) - 1))
        w = (t - native_taus[j - 1]) / (native_taus[j] - native_taus[j - 1])
        out.append((1 - w) * native_q[j - 1] + w * native_q[j])
    return np.stack(out, axis=0)


def quantiles_from_samples(samples: np.ndarray, taus) -> np.ndarray:
    """Empirical quantiles over the sample axis 0."""
    return np.quantile(np.asarray(samples, float), np.asarray(taus), axis=0)
