"""The architecture figures and the exogenous-covariate table agree with the released models.

* every number a figure states in its ``config`` is equal to the model's released configuration, kept as a snapshot in
  data/model_configs/ (taken from the Hugging Face checkpoints on 3 Nov 2026);
* the capability table of Notebooks 05a to 09b says exactly what each adapter class does with covariates.
"""
import json
from pathlib import Path

import pytest

from ch03_zeroshot.architectures import ARCH
from ch03_zeroshot.capabilities import ZERO_SHOT_MODELS
from ch03_zeroshot.models import REGISTRY, model_class

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "data" / "model_configs"


def lookup(cfg, path):
    for part in path.split("."):
        cfg = cfg[part]
    return cfg


@pytest.mark.parametrize("key", [k for k in REGISTRY if ARCH[k]["config"]])
def test_figure_numbers_match_released_config(key):
    snap = json.loads((CONFIGS / f"{key}.json").read_text())
    assert snap["checkpoint"] == model_class(key).checkpoint
    for path, value in ARCH[key]["config"].items():
        assert lookup(snap["config"], path) == value, f"{key}: {path}"


def test_every_model_has_a_checked_figure():
    for key in REGISTRY:
        spec = ARCH[key]
        assert spec["overview"] and spec["complexity"] and len(spec["facts"]) >= 2, key


@pytest.mark.parametrize("row", ZERO_SHOT_MODELS, ids=lambda r: r[0])
def test_capability_table_matches_adapter(row):
    key, _, _, _, mv, past, fut = row[:7]
    cls = model_class(key)
    assert (cls.supports_multivariate, cls.supports_past_covariates, cls.supports_future_covariates) == (mv, past, fut)


def test_capability_table_covers_every_model():
    assert {r[0] for r in ZERO_SHOT_MODELS} == set(REGISTRY)
