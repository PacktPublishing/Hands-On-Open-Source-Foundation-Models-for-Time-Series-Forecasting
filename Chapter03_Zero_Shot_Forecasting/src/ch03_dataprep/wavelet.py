"""FDR-corrected wavelet coherence analysis (Notebook 02).

A numpy/scipy port of biwavelet::wt / wtc / wtc.sig (Gouhier, Grinsted &
Simko; after Torrence & Compo, 1998, and Grinsted, Moore & Jevrejeva, 2004),
with the defaults used by the G7 paper's script, plus:

* Monte Carlo p-values pooled over time inside the cone of influence, and a
  Benjamini-Hochberg correction per scale (replacing the script's
  p = 1 - pchisq(2 r, 2), which can never flag a point above the threshold);
* an explicit phase convention: the FIRST series is the exogenous variable,
  so an arrow pointing up means the exogenous variable leads.
"""
# --- cell: wavelet-transform ---
import numpy as np
from scipy import fft as sfft
from scipy.signal import lfilter

K0 = 6.0  # Morlet omega_0
FOURIER_FACTOR = 4 * np.pi / (K0 + np.sqrt(2 + K0 ** 2))  # period / scale = 1.033


def _wavenumbers(n, dt=1.0):
    k = np.arange(1, n // 2 + 1) * 2 * np.pi / (n * dt)
    return np.concatenate([[0.0], k, -k[: (n - 1) // 2][::-1]])


def wavelet_scales(n_obs, dt=1.0, dj=1 / 12, s0=None, max_scale=None):
    """Scales s_j = s0 2^(j dj), j = 0..J1, as biwavelet::wtc chooses them."""
    s0 = 2 * dt if s0 is None else s0
    max_scale = n_obs * 0.17 * 2 * dt if max_scale is None else max_scale
    j1 = int(round(np.log2(max_scale / s0) / dj))
    return s0 * 2 ** (np.arange(j1 + 1) * dj)


def cone_of_influence(n_obs, dt=1.0):
    """e-folding time sqrt(2) s, in period units, from each edge (biwavelet::wt)."""
    half = np.concatenate([[1e-5], np.arange(1, int(0.5 * (n_obs + 1))),
                           np.arange(int(np.floor(0.5 * n_obs - 1)), 0, -1), [1e-5]])
    return (FOURIER_FACTOR / np.sqrt(2)) * dt * half[:n_obs]


def cwt_morlet(x, scales, dt=1.0):
    """Continuous Morlet wavelet transform by FFT with zero padding.

    W_x(s, t) = sum_k xhat_k psi_hat*(s w_k) e^{i w_k t}, with the normalised
    Morlet psi_hat(s w) = pi^(-1/4) sqrt(s w_1 N) e^{-(s w - w0)^2 / 2} for w > 0.
    x may be 1-D (n,) or 2-D (batch, n); returns (scales, n) or (batch, scales, n).
    """
    x = np.atleast_2d(np.asarray(x))
    real = np.float32 if x.dtype == np.float32 else np.float64
    x = x.astype(real)
    n_obs = x.shape[-1]
    x = x - x.mean(axis=-1, keepdims=True)
    npad = int(2 ** np.ceil(np.log2(n_obs) + 1))
    k = _wavenumbers(npad, dt)
    f = sfft.fft(x, n=npad, axis=-1)
    expnt = -((scales[:, None] * k[None, :] - K0) ** 2) / 2 * (k > 0)
    norm = np.sqrt(scales[:, None] * k[1]) * np.pi ** (-0.25) * np.sqrt(npad)
    daughter = (norm * np.exp(expnt) * (k > 0)).astype(real)
    wave = sfft.ifft(f[:, None, :] * daughter[None, :, :], axis=-1, workers=-1)[..., :n_obs]
    return wave[0] if wave.shape[0] == 1 else wave


def smooth_wavelet(w, scales, dt=1.0, dj=1 / 12):
    """Smoothing operator S of Torrence & Webster (1999), as biwavelet::smooth.wavelet.

    Time: Gaussian exp(-t^2 / (2 s^2)) applied in Fourier space. Scale: boxcar of
    width 0.6 (0.6 / dj scales) with fractional end weights. Works on (scales, n)
    or (batch, scales, n).
    """
    w = np.asarray(w)
    m = w.shape[-1]
    npad = int(2 ** np.ceil(np.log2(m)))
    if np.iscomplexobj(w):
        k = np.arange(1, npad // 2 + 1) * 2 * np.pi / npad
        k = np.concatenate([[0.0], k, -k[: int(0.5 * (npad - 1))][::-1]])
        g = np.exp(-0.5 * (scales[:, None] / dt) ** 2 * k[None, :] ** 2).astype(w.real.dtype)
        tw = sfft.ifft(g * sfft.fft(w, n=npad, axis=-1), axis=-1, workers=-1)[..., :m]
    else:  # real input: same filter with the real FFT (half the work)
        k = np.arange(npad // 2 + 1) * 2 * np.pi / npad
        g = np.exp(-0.5 * (scales[:, None] / dt) ** 2 * k[None, :] ** 2).astype(w.dtype)
        tw = sfft.irfft(g * sfft.rfft(w, n=npad, axis=-1), n=npad, axis=-1, workers=-1)[..., :m]
    # biwavelet: full convolution along the scale axis with the boxcar, then keep
    # rows floor(L/2) .. floor(L/2) + n_scales - 1; written as a banded matrix.
    return np.matmul(_scale_matrix(len(scales), dj).astype(tw.real.dtype), tw)


def _scale_matrix(n_scales, dj):
    d0 = 0.3 / dj
    frac = d0 % 1
    length = 2 * int(round(d0))
    ker = np.concatenate([[frac], np.ones(length - 1), [frac]])
    ker = ker / (length - 1 + 2 * frac)
    h = len(ker) // 2
    mat = np.zeros((n_scales, n_scales))
    for row in range(n_scales):
        for k, wk in enumerate(ker):
            col = row + h - k
            if 0 <= col < n_scales:
                mat[row, col] = wk
    return mat


def coherence(wx, wy, scales, dt=1.0, dj=1 / 12):
    """Squared wavelet coherence R^2 = |S(W_xy / s)|^2 / (S(|W_x|^2 / s) S(|W_y|^2 / s))."""
    sinv = (1 / scales)[:, None]
    sxy = smooth_wavelet(sinv * wx * np.conj(wy), scales, dt, dj)
    sxx = smooth_wavelet(sinv * np.abs(wx) ** 2, scales, dt, dj)
    syy = smooth_wavelet(sinv * np.abs(wy) ** 2, scales, dt, dj)
    return np.abs(sxy) ** 2 / (sxx * syy)


# --- cell: wavelet-significance ---
def ar1_coefficient(x):
    """Lag-1 autoregressive coefficient by exact Gaussian ML (R arima(order = c(1,0,0)))."""
    from statsmodels.tsa.arima.model import ARIMA

    import warnings

    x = np.asarray(x, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phi = float(ARIMA(x, order=(1, 0, 0)).fit().params[1])
    return float(np.clip(phi, -0.999, 0.999))  # deterministic inputs give phi -> 1


def ar1_surrogates(phi, n, size, rng):
    """`size` AR(1) series of length n with burn-in, as biwavelet::ar1_ma0_sim."""
    burn = 2 + int(np.ceil(6 / np.log(1 / max(abs(phi), 1e-6))))
    e = rng.standard_normal((size, n + burn))
    return lfilter([1.0], [1.0, -phi], e, axis=-1)[:, burn:].astype(np.float32)


def coi_mask(period, coi):
    """True where (scale, time) is inside the cone of influence (period <= coi)."""
    return period[:, None] <= coi[None, :]


def benjamini_hochberg(p, alpha=0.10):
    """Boolean mask of hypotheses rejected by Benjamini & Hochberg (1995)."""
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = alpha * np.arange(1, m + 1) / m
    below = p[order] <= thresh
    keep = np.zeros(m, bool)
    if below.any():
        kmax = np.max(np.nonzero(below)[0])
        keep[order[: kmax + 1]] = True
    return keep


def wavelet_coherence(x, y, dt=1.0, dj=1 / 12, n_surrogates=1000, alpha=0.10,
                      seed=2026, bins=4000, batch=None, mem_bytes=1e8):
    """FDR-corrected wavelet coherence of x (exogenous, first) and y (target).

    Returns a dict with rsq, phase (arg W_x W_y*, unsmoothed as in biwavelet),
    period, coi, inside (COI mask), p (Monte Carlo p-values), sig (BH-FDR mask),
    sig95_pointwise (biwavelet-style 95 % surrogate quantile per scale) and the
    AR(1) coefficients used for the surrogates.

    Monte Carlo p-value at (s, t): p = (1 + #{null R^2 at scale s >= R^2(s, t)}) / (1 + N),
    the null pooled over all surrogates and all time points inside the COI.
    The pooled null is kept as a histogram with `bins` bins per scale. The
    surrogates run in single precision (coherence needs no more), the observed
    map in double precision.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    scales = wavelet_scales(n, dt, dj)
    period = FOURIER_FACTOR * scales
    coi = cone_of_influence(n, dt)
    inside = coi_mask(period, coi)
    wx, wy = cwt_morlet(x, scales, dt), cwt_morlet(y, scales, dt)
    rsq = coherence(wx, wy, scales, dt, dj)
    phase = np.angle(wx * np.conj(wy))

    rng = np.random.default_rng(seed)
    phi_x, phi_y = ar1_coefficient(x), ar1_coefficient(y)
    edges = np.linspace(0, 1, bins + 1)
    hist = np.zeros((len(scales), bins), np.int64)
    n_inside = inside.sum(axis=1)
    npad = int(2 ** np.ceil(np.log2(n) + 1))
    batch = batch or max(1, int(mem_bytes // (len(scales) * npad * 8)))
    done = 0
    while done < n_surrogates:
        b = min(batch, n_surrogates - done)
        sx = cwt_morlet(ar1_surrogates(phi_x, n, b, rng), scales, dt)
        sy = cwt_morlet(ar1_surrogates(phi_y, n, b, rng), scales, dt)
        if sx.ndim == 2:
            sx, sy = sx[None], sy[None]
        null = coherence(sx, sy, scales, dt, dj)  # (b, scales, n)
        jj = np.broadcast_to(np.arange(len(scales))[None, :, None], null.shape)[:, inside]
        kk = np.minimum((np.clip(null[:, inside], 0, 1) * bins).astype(np.int64), bins - 1)
        hist += np.bincount((jj * bins + kk).ravel(), minlength=len(scales) * bins).reshape(len(scales), bins)
        done += b

    # survival counts per bin edge: number of null values >= edge
    surv = np.concatenate([np.cumsum(hist[:, ::-1], axis=1)[:, ::-1],
                           np.zeros((len(scales), 1), np.int64)], axis=1)
    p = np.full(rsq.shape, np.nan)
    sig = np.zeros(rsq.shape, bool)
    sig95 = np.full(len(scales), np.nan)
    for j in range(len(scales)):
        total = n_surrogates * n_inside[j]
        if total == 0:
            continue
        r = np.clip(rsq[j], 0, 1)
        pos = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, bins - 1)
        frac = (r - edges[pos]) / (edges[1] - edges[0])
        # interpolate inside the bin (null assumed uniform within a bin)
        count = surv[j, pos + 1] + (1 - frac) * hist[j, pos]
        p[j] = (1 + count) / (1 + total)
        cdf = np.cumsum(hist[j]) / total
        sig95[j] = edges[1:][min(np.searchsorted(cdf, 0.95), bins - 1)]
        idx = np.nonzero(inside[j])[0]
        sig[j, idx] = benjamini_hochberg(p[j, idx], alpha)
    return {"rsq": rsq, "phase": phase, "period": period, "scales": scales, "coi": coi,
            "inside": inside, "p": p, "sig": sig, "sig95_pointwise": sig95,
            "phi_x": phi_x, "phi_y": phi_y, "n_surrogates": n_surrogates, "alpha": alpha}


# --- cell: wavelet-summary ---
def summarise_coherence(res, times, bands):
    """Turn one coherence map into numbers.

    bands: list of (label, low_period, high_period) in time units of the data.
    For each band: share of the COI area that is FDR-significant, mean
    coherence and circular-mean phase inside the significant area, and the
    implied lead of the exogenous variable (lead_fraction(phase) x the
    R^2-weighted geometric mean period of the significant points; positive =
    exogenous variable leads).
    """
    rows = []
    period, inside, sig = res["period"], res["inside"], res["sig"]
    for label, lo, hi in bands:
        rmask = (period >= lo) & (period < hi)
        area = inside[rmask].sum()
        s = sig[rmask] & inside[rmask]
        share = s.sum() / area if area else np.nan
        if s.any():
            ph = res["phase"][rmask][s]
            w = res["rsq"][rmask][s]
            mean_phase = np.angle(np.sum(w * np.exp(1j * ph)))
            per = np.broadcast_to(period[rmask][:, None], s.shape)[s]
            centre = np.exp(np.sum(w * np.log(per)) / np.sum(w))
            rows.append({"band": label, "sig_share": share,
                         "mean_coh_sig": float(w.mean()), "centre_period": float(centre),
                         "phase_deg": float(np.degrees(mean_phase)),
                         "lead_of_exog": float(lead_fraction(np.degrees(mean_phase)) * centre),
                         "reading": phase_reading(np.degrees(mean_phase))})
        else:
            rows.append({"band": label, "sig_share": share, "mean_coh_sig": np.nan,
                         "centre_period": np.nan, "phase_deg": np.nan, "lead_of_exog": np.nan,
                         "reading": "no significant area"})
    return rows


def lead_fraction(phase_deg):
    """Lead of the exogenous series in cycles, reading the phase as Aguiar-Conraria & Soares (2014).

    phi in (0, 90): in phase, exogenous leads;   phi in (-90, 0): in phase, target leads;
    phi in (90, 180): anti-phase, target leads;  phi in (-180, -90): anti-phase, exogenous leads.
    For anti-phase angles the lead is measured against the inverted series.
    """
    a = (phase_deg + 180) % 360 - 180
    return a / 360 if abs(a) <= 90 else (a - 180 * np.sign(a)) / 360


def phase_reading(phase_deg, tol=10):
    """Plain-language reading of a mean phase angle (exogenous series first)."""
    if phase_deg is None or np.isnan(phase_deg):
        return "no significant area"
    a = (phase_deg + 180) % 360 - 180
    if abs(a) <= tol:
        return "in phase, no clear lead"
    if abs(a) >= 180 - tol:
        return "anti-phase, no clear lead"
    relation = "in phase" if abs(a) < 90 else "anti-phase"
    leader = "exogenous leads" if lead_fraction(a) > 0 else "target leads"
    return f"{relation}, {leader}"


def episode_years(res, times, top=3):
    """Years with the largest share of significant area (inside the COI)."""
    import pandas as pd

    years = pd.DatetimeIndex(times).year
    s = (res["sig"] & res["inside"]).sum(axis=0)
    a = res["inside"].sum(axis=0)
    df = pd.DataFrame({"year": years, "sig": s, "area": a}).groupby("year").sum()
    df = df[df["area"] > 0]
    share = (df["sig"] / df["area"]).sort_values(ascending=False)
    return [int(y) for y in share.index[:top] if share[y] > 0]


# --- cell: wavelet-calibration ---
def band_share(res, lo, hi):
    """Share of the COI area with periods in [lo, hi) that is FDR-significant."""
    rmask = (res["period"] >= lo) & (res["period"] < hi)
    area = res["inside"][rmask].sum()
    return float((res["sig"][rmask] & res["inside"][rmask]).sum() / area) if area else np.nan


def null_best_shares(phi_x, phi_y, n, bands, shortest_horizon, n_pairs, n_surrogates=200, alpha=0.10, seed=0):
    """The verdict rule's statistic under independence.

    For n_pairs independent AR(1) pairs with coefficients (phi_x, phi_y) and
    length n, run the same coherence test and return, for each pair, the
    largest FDR-significant share among the bands that reach the horizons
    (upper period >= shortest_horizon). Its upper quantiles are the
    thresholds a real pair must beat.
    """
    rng = np.random.default_rng(seed)
    best = []
    for _ in range(n_pairs):
        a = ar1_surrogates(phi_x, n, 1, rng)[0].astype(float)
        b = ar1_surrogates(phi_y, n, 1, rng)[0].astype(float)
        res = wavelet_coherence(a, b, n_surrogates=n_surrogates, alpha=alpha, seed=int(rng.integers(2**31)))
        best.append(np.nanmax([band_share(res, lo, hi) for _, lo, hi in bands if hi >= shortest_horizon]))
    return np.array(best)


def calibrated_thresholds(null_best, keep_q=0.95, weak_q=0.80, keep_floor=0.10, weak_floor=0.05):
    """keep / weak thresholds: the null's 95th and 80th percentiles, never below 10 % and 5 %."""
    return {"keep": max(keep_floor, float(np.quantile(null_best, keep_q))),
            "weak": max(weak_floor, float(np.quantile(null_best, weak_q)))}

