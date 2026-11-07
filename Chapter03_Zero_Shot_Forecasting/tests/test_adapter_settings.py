"""Settings the zero-shot runner passes to individual models (no weights are downloaded)."""
from ch03_zeroshot.data import SEASON, YEAR_STEPS
from ch03_zeroshot.models import model_class


def test_every_dataset_has_a_year_length():
    assert set(YEAR_STEPS) == set(SEASON)


def test_flowstate_scale_factor_follows_the_model_card():
    # model card, "Temporal Scaling": hourly 1.0, daily without a weekly cycle 0.0656, monthly 2
    fs = model_class("flowstate")()
    for steps, card in ((24, 1.0), (365, 0.0656), (12, 2.0)):
        fs.set_season_length(steps)
        assert abs(fs.scale_factor - card) < 1e-3


def test_models_without_a_time_scale_ignore_it():
    m = model_class("chronos-bolt")()
    m.set_season_length(365)
    assert not hasattr(m, "season_length")


def test_timesfm25_covariate_step_runs_jax_on_the_cpu():
    # On a Colab GPU runtime the pinned jax does not match Colab's JAX CUDA plugin ("No FFI handler registered
    # for cusolver_..._ffi"), so TimesFM-2.5's XReg step must not touch the GPU.
    import pytest

    jax = pytest.importorskip("jax")
    from ch03_zeroshot.models.timesfm import _jax_on_cpu

    _jax_on_cpu()
    assert jax.config.jax_platforms == "cpu"
    assert {d.platform for d in jax.devices()} == {"cpu"}
