"""D2 covariates: day length, Fourier terms, causal Haar MODWT."""
import numpy as np
import pandas as pd
import pytest

from ch03_dataprep import features as ft


def test_day_length_solstices_at_52_5_north():
    # Almanac values for about 52.5 N (sunrise/sunset with refraction): about 16 h 50 m in June, 7 h 40 m in December
    d = ft.day_length_hours(pd.to_datetime(["2025-06-21", "2025-12-21", "2025-03-20"]))
    assert d.iloc[0] == pytest.approx(16.83, abs=0.2)
    assert d.iloc[1] == pytest.approx(7.66, abs=0.2)
    assert d.iloc[2] == pytest.approx(12.1, abs=0.2)  # just over 12 h at the equinox (refraction)


def test_fourier_terms_are_annual():
    idx = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    f = ft.fourier_terms(idx, 3)
    assert f.shape[1] == 6
    assert np.allclose(f.loc["2023-03-01"], f.loc["2025-03-01"])
    assert f.abs().max().max() <= 1


def test_modwt_is_causal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    base = ft.wavelet_bands(pd.Series(x))
    x2 = x.copy()
    x2[1500:] += 10  # change the future only
    moved = ft.wavelet_bands(pd.Series(x2))
    pd.testing.assert_frame_equal(base.iloc[:1500], moved.iloc[:1500])
    assert not np.allclose(base.iloc[1500].to_numpy(), moved.iloc[1500].to_numpy())


def test_modwt_bands_add_up_to_the_series():
    # Haar MODWT: X_t = sum_j W_j,t + V_J,t once the start-up rows are dropped
    x = np.random.default_rng(1).normal(size=1000)
    b = ft.wavelet_bands(pd.Series(x)).dropna()
    assert np.allclose(b.sum(axis=1).to_numpy(), x[b.index])
    assert b.index[0] == 255
