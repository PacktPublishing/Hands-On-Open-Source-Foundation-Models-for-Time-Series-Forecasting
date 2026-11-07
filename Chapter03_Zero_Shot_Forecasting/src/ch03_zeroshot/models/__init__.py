"""Model registry: ``get_model("chronos2")`` returns a ready adapter.

Each entry names the environment it runs in (``requirements/env-<name>.txt``), because some
libraries pin incompatible versions of numpy, torch or transformers. Imports are lazy, so
importing the registry never needs a model library.
"""
from __future__ import annotations

import importlib

from .base import QUANTILES, CannotForecast, ForecastOutput, SeriesInput, ZeroShotForecaster  # noqa: F401

# key: (module, class, default kwargs, environment, notebook)
REGISTRY = {
    "chronos2":     ("chronos", "Chronos2", {}, "core", "05a"),
    "chronos-bolt": ("chronos", "ChronosBolt", {}, "core", "05a"),
    "timesfm25":    ("timesfm", "TimesFM25", {}, "core", "05a"),
    "timesfm3":     ("timesfm", "TimesFM3", {}, "core", "05a"),
    "tirex":        ("tirex", "TiRex", {}, "core", "05a"),
    "tirex2":       ("tirex", "TiRex2", {}, "core", "05a"),
    "timestral":    ("timestral", "Timestral", {}, "core", "05b"),
    "timestral-5m": ("timestral", "TimestralSmall", {}, "core", "05b"),
    "ttm":          ("ibm", "TTM", {}, "core", "05b"),
    "flowstate":    ("ibm", "FlowState", {}, "core", "05b"),
    "patchtst-fm":  ("ibm", "PatchTSTFM", {}, "core", "05b"),
    "t0":           ("others", "T0", {}, "core", "05b"),
    "tabpfn-ts":    ("others", "TabPFNTS", {"n_estimators": 2}, "core", "05b"),
    "moirai11":     ("moirai", "Moirai11", {}, "moirai", "06"),
    "moirai-moe":   ("moirai", "MoiraiMoE", {}, "moirai", "06"),
    "moirai2":      ("moirai", "Moirai2", {}, "moirai", "06"),
    "toto2":        ("toto", "Toto2", {}, "toto2", "07a"),
    "toto1":        ("toto", "Toto1", {"num_samples": 32, "samples_per_batch": 32}, "toto1", "07b"),
    "sundial":      ("thuml", "Sundial", {"num_samples": 20}, "thuml", "08"),
    "timer":        ("thuml", "Timer", {}, "thuml", "08"),
    "timemoe":      ("thuml", "TimeMoE", {}, "thuml", "08"),
    "kairos":       ("kairos", "Kairos", {}, "kairos", "09a"),
    "lag-llama":    ("lagllama", "LagLlama", {}, "lagllama", "09b"),
}


def get_model(key: str, device: str = "cpu", **kwargs) -> ZeroShotForecaster:
    mod, cls, defaults, _env, _nb = REGISTRY[key]
    module = importlib.import_module(f"{__name__}.{mod}")
    return getattr(module, cls)(device=device, **{**defaults, **kwargs})


def model_class(key: str):
    mod, cls, *_ = REGISTRY[key]
    return getattr(importlib.import_module(f"{__name__}.{mod}"), cls)


def environment_of(key: str) -> str:
    return REGISTRY[key][3]


def models_in_notebook(nb: str) -> list[str]:
    return [k for k, v in REGISTRY.items() if v[4] == nb]
