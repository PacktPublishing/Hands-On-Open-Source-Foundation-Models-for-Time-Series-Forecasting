"""Wavelet coherence: phase convention, null calibration and helpers (NB02 runs the same checks)."""
import numpy as np
import pytest

from ch03_dataprep import wavelet as wv


def test_phase_up_when_first_series_leads():
    # x leads y by a quarter period at period 40: the arrow must point up (+90 degrees)
    rng = np.random.default_rng(0)
    t = np.arange(400)
    x = np.sin(2 * np.pi * t / 40) + 0.1 * rng.normal(size=400)
    y = np.sin(2 * np.pi * (t - 10) / 40) + 0.1 * rng.normal(size=400)
    res = wv.wavelet_coherence(x, y, n_surrogates=50)
    j = np.argmin(abs(res["period"] - 40))
    mid = slice(100, 300)
    phase = np.degrees(np.angle(np.mean(np.exp(1j * res["phase"][j, mid]))))
    assert 80 < phase < 100
    assert res["rsq"][j, mid].mean() > 0.95
    assert wv.phase_reading(phase) == "in phase, exogenous leads"


def test_null_is_calibrated():
    # two independent AR(1) series: about 5 % pointwise p < 0.05, almost nothing after FDR
    rng = np.random.default_rng(3)
    x = wv.ar1_surrogates(0.5, 400, 1, rng)[0].astype(float)
    y = wv.ar1_surrogates(0.3, 400, 1, rng)[0].astype(float)
    res = wv.wavelet_coherence(x, y, n_surrogates=300, seed=7)
    p_inside = res["p"][res["inside"]]
    assert 0.02 < np.mean(p_inside < 0.05) < 0.10
    assert np.mean(res["sig"][res["inside"]]) < 0.03


@pytest.mark.parametrize("phase, expected", [
    (45, "in phase, exogenous leads"), (-45, "in phase, target leads"),
    (135, "anti-phase, target leads"), (-135, "anti-phase, exogenous leads"),
    (0, "in phase, no clear lead"), (179, "anti-phase, no clear lead")])
def test_phase_reading_quadrants(phase, expected):
    assert wv.phase_reading(phase) == expected


def test_lead_fraction_signs():
    assert wv.lead_fraction(90) == pytest.approx(0.25)
    assert wv.lead_fraction(-90) == pytest.approx(-0.25)
    assert wv.lead_fraction(135) < 0 < wv.lead_fraction(-135)


def test_benjamini_hochberg_matches_definition():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    # BH at 5 %: largest k with p_(k) <= k/m * alpha is k = 2
    assert wv.benjamini_hochberg(p, alpha=0.05).sum() == 2


def test_period_is_about_1_03_scale():
    assert wv.FOURIER_FACTOR == pytest.approx(1.033, abs=1e-3)
