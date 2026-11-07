"""Covariates for D2 (Notebook 00): Fourier terms, day length, causal wavelet bands."""
# --- cell: features ---
import numpy as np
import pandas as pd


def fourier_terms(index, k_max=3):
    """Annual Fourier terms sin/cos(2 pi k d / L), d = day of year, L = days in year.

    Deterministic, so they are known for any future date (known-future class).
    """
    idx = pd.DatetimeIndex(index)
    doy = idx.dayofyear.to_numpy() - 1
    year_len = np.where(idx.is_leap_year, 366.0, 365.0)
    out = {}
    for k in range(1, k_max + 1):
        ang = 2 * np.pi * k * doy / year_len
        out[f"fourier_sin_{k}"] = np.sin(ang)
        out[f"fourier_cos_{k}"] = np.cos(ang)
    return pd.DataFrame(out, index=idx)


def day_length_hours(index, latitude=52.5, p=0.8333):
    """Day length (hours) from the CBM model of Forsythe et al. (1995).

    theta = 0.2163108 + 2 atan(0.9671396 tan(0.00860 (J - 186)))   revolution angle
    phi   = asin(0.39795 cos(theta))                                 solar declination
    D     = 24 - (24 / pi) acos[(sin(p pi/180) + sin(L pi/180) sin(phi))
                                / (cos(L pi/180) cos(phi))]
    p = 0.8333 degrees: sunrise and sunset when the top of the sun is level
    with the horizon, including refraction (the usual almanac definition).
    """
    idx = pd.DatetimeIndex(index)
    j = idx.dayofyear.to_numpy().astype(float)
    theta = 0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.00860 * (j - 186)))
    phi = np.arcsin(0.39795 * np.cos(theta))
    lat = np.deg2rad(latitude)
    arg = (np.sin(np.deg2rad(p)) + np.sin(lat) * np.sin(phi)) / (np.cos(lat) * np.cos(phi))
    d = 24 - (24 / np.pi) * np.arccos(np.clip(arg, -1, 1))
    return pd.Series(d, index=idx, name="daylength_h")


def causal_haar_modwt(x, levels=8):
    """Causal Haar MODWT coefficients W_1..W_J and scaling coefficients V_J.

    Pyramid algorithm (Percival & Walden, 2000, ch. 5) with the Haar MODWT
    filters h = (1/2, -1/2), g = (1/2, 1/2):
        W_{j,t} = (V_{j-1,t} - V_{j-1,t-2^(j-1)}) / 2
        V_{j,t} = (V_{j-1,t} + V_{j-1,t-2^(j-1)}) / 2,   V_0 = X.
    Each value uses only X_t, X_{t-1}, ... (no future values), so it is a
    legitimate past-only covariate. The first 2^J - 1 values depend on the
    start of the record and are set to NaN (computed on the full record, they
    fall long before the 10,000-day window).
    """
    v = np.asarray(x, float)
    n = len(v)
    out = {}
    for j in range(1, levels + 1):
        lag = 2 ** (j - 1)
        shifted = np.full(n, np.nan)
        shifted[lag:] = v[:-lag]
        out[f"W{j}"] = (v - shifted) / 2
        v = (v + shifted) / 2
    out[f"V{levels}"] = v
    width = 2 ** levels - 1
    for key in out:
        out[key][:width] = np.nan
    return out


def wavelet_bands(series, levels=8):
    """Five causal band features from the Haar MODWT coefficients.

    wav_d1_2: W1 + W2 (periods of about 2 to 8 days), wav_d3_4: W3 + W4 (8 to 32),
    wav_d5_6: W5 + W6 (32 to 128), wav_d7_8: W7 + W8 (128 to 512), wav_s8: V8 (> 512).
    """
    c = causal_haar_modwt(np.asarray(series, float), levels)
    bands = {
        "wav_d1_2": c["W1"] + c["W2"], "wav_d3_4": c["W3"] + c["W4"],
        "wav_d5_6": c["W5"] + c["W6"], "wav_d7_8": c["W7"] + c["W8"],
        "wav_s8": c[f"V{levels}"],
    }
    return pd.DataFrame(bands, index=getattr(series, "index", None))
