"""Forecastability measures behave as expected on series with known structure."""
import numpy as np

from ch03_dataprep import forecastability as fc


def _ar1(phi, n, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_white_noise_is_unforecastable():
    x = np.random.default_rng(0).normal(size=3000)
    assert abs(fc.kaboudan_eta(x, 7, 20)["eta"]) < 0.1
    assert fc.entropy_measures(x)["foreca_omega"] < 0.05
    assert fc.naive_skill(x, 1) < -0.8          # naive is worse than the mean (NSE about -1)


def test_persistent_series_is_forecastable_at_short_horizons():
    x = _ar1(0.9, 3000)
    assert fc.ar_variance_explained(x, 20)["var_explained"] > 0.7
    assert fc.kaboudan_eta(x, 1, 20)["eta"] > 0.6
    assert fc.kaboudan_eta(x, 1, 20)["eta"] > fc.kaboudan_eta(x, 20, 20)["eta"]


def test_variance_ratio_random_walk_near_one():
    rw = np.cumsum(np.random.default_rng(2).normal(size=3000))
    out = fc.variance_ratio(rw, 5)
    assert 0.85 < out["vr"] < 1.15 and out["p"] > 0.01


def test_seasonal_naive_beats_naive_on_a_pure_cycle():
    t = np.arange(1200)
    x = np.sin(2 * np.pi * t / 12)
    assert fc.naive_skill(x, 6, season=12) > 0.99 > fc.naive_skill(x, 6)
