"""Checks of the Granger, SURD and rolling-correlation helpers used in Notebook 02."""
import numpy as np
import pandas as pd
from scipy import stats

from ch03_dataprep.causality import (_standardise, diks_panchenko, granger_linear, phase_randomise,
                                     rolling_spearman, scale_bands, surd_shares)


def test_diks_panchenko_matches_the_formula():
    rng = np.random.default_rng(0)
    x, y, eps = rng.standard_normal(150), rng.standard_normal(150), 1.0
    xs, ys = _standardise(x), _standardise(y)
    X, Y, Z = xs[:-1], ys[:-1], ys[1:]
    n = len(Z)
    near = lambda v: np.abs(v[:, None] - v[None, :]) <= eps
    ix, iy, iz = near(X), near(Y), near(Z)
    np.fill_diagonal(iy, False)
    f = lambda a, d: a.sum(axis=1) / ((n - 1) * (2 * eps) ** d)
    t_brute = np.mean(f(ix & iy & iz, 3) * f(iy, 1) - f(ix & iy, 2) * f(iy & iz, 2))
    assert np.isclose(diks_panchenko(x, y, eps=eps)["T"], t_brute, rtol=1e-10)


def test_linear_granger_finds_a_strong_link():
    rng = np.random.default_rng(1)
    x, e = rng.standard_normal(400), rng.standard_normal(400)
    y = np.zeros(400)
    for t in range(1, 400):
        y[t] = 0.4 * y[t - 1] + 0.5 * x[t - 1] + e[t]
    assert granger_linear(y, x, 4)["p_value"] < 1e-6
    assert granger_linear(x, y, 4)["p_value"] > 0.001


def test_phase_randomise_keeps_mean_and_spectrum():
    rng = np.random.default_rng(2)
    v = np.cumsum(rng.standard_normal(301)) * 0.1 + 5
    s = phase_randomise(v, rng)
    assert np.isclose(s.mean(), v.mean())
    assert np.allclose(np.abs(np.fft.rfft(s - s.mean())), np.abs(np.fft.rfft(v - v.mean())))


def test_scale_bands_are_causal():
    rng = np.random.default_rng(3)
    v = rng.standard_normal(300)
    w = v.copy()
    w[200:] += 10.0                                   # change only the future
    a, b = scale_bands(v, "d"), scale_bands(w, "d")
    for k in a:
        assert np.allclose(a[k][:200], b[k][:200], equal_nan=True)


def test_surd_toy_systems():
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal(20_000), rng.standard_normal(20_000)
    target = np.roll(a, 1) + 0.3 * rng.standard_normal(20_000)
    copy = surd_shares(target, {"a": a, "b": a + 1e-9 * b}, lag=1, bins=4)
    assert copy["redundant a+b"] > 0.4 and copy["unique a"] < 0.01 and copy["unique b"] < 0.01
    single = surd_shares(target, {"a": a, "b": b}, lag=1, bins=4)
    assert single["unique a"] > 0.4 and single["unique b"] < 0.01
    assert np.isclose(sum(single.values()), 1.0)
    product = surd_shares(np.roll(a * b, 1), {"a": a, "b": b}, lag=1, bins=2)
    assert product["synergy a+b"] > 0.9


def test_rolling_spearman_matches_scipy():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    y, x = pd.Series(rng.standard_normal(120), idx), pd.Series(rng.standard_normal(120), idx)
    rho, band = rolling_spearman(y, x, 60, lag=1)
    expected = stats.spearmanr(y.iloc[1:61], x.shift(1).iloc[1:61]).statistic
    assert np.isclose(rho.iloc[0], expected)
    assert np.isclose(band, 1.96 * np.sqrt(1.06 / 57))
