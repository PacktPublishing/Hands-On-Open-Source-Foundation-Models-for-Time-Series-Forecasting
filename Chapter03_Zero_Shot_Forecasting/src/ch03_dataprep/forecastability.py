"""Forecastability measures (Notebook 03), following Chapter 2 of Manokhin,
*Mastering Modern Time Series Forecasting*, with each measure cited to its
original paper in the notebook."""
# --- cell: forecastability-linear ---
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


def adi_cv2(x):
    """Syntetos-Boylan demand classes: ADI = n / n_nonzero and CV^2 of non-zero values.

    Cut-offs ADI = 1.32 and CV^2 = 0.49 (Syntetos, Boylan & Croston, 2005).
    Only defined for non-negative series; returns None otherwise.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if np.any(x < 0):
        return None
    nz = x[x > 0]
    adi = len(x) / len(nz)
    cv2 = (nz.std(ddof=1) / nz.mean()) ** 2
    cls = ("smooth" if adi < 1.32 and cv2 < 0.49 else "erratic" if adi < 1.32
           else "intermittent" if cv2 < 0.49 else "lumpy")
    return {"adi": adi, "cv2": cv2, "class": cls}


def acf_at(x, lags):
    """Autocorrelation at the given lags."""
    x = np.asarray(x, float)
    r = acf(x[~np.isnan(x)], nlags=max(lags), fft=True)
    return {lag: float(r[lag]) for lag in lags}


def ar_variance_explained(x, max_lag):
    """1 - sigma_e^2 / sigma_y^2 of the AIC-best AR(p), p <= max_lag (OLS fit)."""
    from statsmodels.tsa.ar_model import ar_select_order

    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = ar_select_order(x, maxlag=max_lag, ic="aic", trend="c")
        res = sel.model.fit()
    p = len(sel.ar_lags or [])
    return {"ar_order": p, "var_explained": float(1 - np.var(res.resid) / np.var(x[p:]))}


def naive_skill(x, horizon, season=None):
    """Nash-Sutcliffe efficiency of the naive (or seasonal-naive) H-step forecast.

    NSE = 1 - sum (y_{t+H} - yhat)^2 / sum (y_{t+H} - mean(y))^2, with
    yhat = y_t (naive) or y_{t+H-m*ceil(H/m)} (seasonal naive, period m).
    NSE = 1 is perfect, 0 is no better than the mean, < 0 is worse than the mean.
    """
    x = np.asarray(x, float)
    back = horizon if season is None else season * int(np.ceil(horizon / season))
    pred, obs = x[: len(x) - back], x[back:]
    ok = ~np.isnan(pred) & ~np.isnan(obs)
    return float(1 - np.sum((obs[ok] - pred[ok]) ** 2) / np.sum((obs[ok] - obs[ok].mean()) ** 2))


def variance_ratio(x, k):
    """Lo-MacKinlay variance ratio VR(k) with the heteroskedasticity-robust test.

    VR(k) = Var(y_{t+k} - y_t) / (k Var(y_{t+1} - y_t)). About 1 for a random
    walk, below 1 for mean reversion, above 1 for trending (momentum).
    """
    from arch.unitroot import VarianceRatio

    x = np.asarray(x, float)
    vr = VarianceRatio(x[~np.isnan(x)], lags=k, robust=True)
    return {"vr": float(vr.vr), "p": float(vr.pvalue)}


def kaboudan_eta(x, horizon, max_lag, n_shuffles=5, seed=0):
    """Kaboudan (1999) shuffle score at horizon H: eta = 1 - SSE(original) / SSE(shuffled).

    A direct H-step linear model, y_{t+H} = c + sum_{j<p} b_j y_{t-j} + e, with
    p chosen by AIC (at most max_lag), is fitted to the whole series, and the
    same model is fitted to random permutations of it (the shuffled SSE is
    averaged over n_shuffles permutations). As in Kaboudan's definition, both
    fits use the whole series, so the original and the shuffled copies hold
    exactly the same values and differ only in their order. (Scoring the
    original on its last stretch and the shuffles on a random mix would also
    compare a calm or volatile final period with the average, which is not
    order.) Shuffling destroys temporal order but keeps the distribution, so
    eta near 0 means no order this model class can exploit at horizon H, and
    eta near 1 means most of the error disappears thanks to order. Kaboudan
    used genetic programming; a linear model keeps the idea simple and fast.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    p = max(1, ar_variance_explained(x, max_lag)["ar_order"])

    def sse(z):
        n = len(z) - horizon - p + 1
        X = np.column_stack([np.ones(n)] + [z[p - 1 - j: p - 1 - j + n] for j in range(p)])
        y = z[p - 1 + horizon: p - 1 + horizon + n]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(np.sum((y - X @ beta) ** 2))

    rng = np.random.default_rng(seed)
    sse_shuffled = np.mean([sse(rng.permutation(x)) for _ in range(n_shuffles)])
    return {"eta": 1 - sse(x) / sse_shuffled, "ar_order": p}


# --- cell: forecastability-entropy ---
def entropy_measures(x, order=4):
    """Spectral entropy / ForeCA Omega, permutation, sample and SVD entropy.

    All normalised to [0, 1] where possible; lower entropy = more structure.
    Omega = 1 - H_spectral (Goerg, 2013): the share of the spectrum that is
    not flat, 0 for white noise and 1 for a pure sinusoid.
    """
    import antropy as ant

    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    h_spec = float(ant.spectral_entropy(x, sf=1.0, method="welch", normalize=True,
                                        nperseg=min(len(x), 512)))
    return {
        "spectral_entropy": h_spec, "foreca_omega": 1 - h_spec,
        "perm_entropy": float(ant.perm_entropy(x, order=order, delay=1, normalize=True)),
        "sample_entropy": float(ant.sample_entropy(x, order=2)),
        "svd_entropy": float(ant.svd_entropy(x, order=order, delay=1, normalize=True)),
    }


def multiscale_entropy(x, scales=(1, 2, 5, 10, 20)):
    """Sample entropy of coarse-grained series (Costa, Goldberger & Peng, 2002).

    The tolerance r = 0.2 SD is fixed from the original series, as in Costa et al.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    r = 0.2 * x.std(ddof=0)
    out = {}
    for s in scales:
        n = len(x) // s
        cg = x[: n * s].reshape(n, s).mean(axis=1)
        out[s] = float(_sampen_fixed_r(cg, 2, r)) if n > 50 else np.nan
    return out


def _sampen_fixed_r(x, m, r):
    """Sample entropy with an absolute tolerance r (Richman & Moorman, 2000)."""
    x = np.asarray(x, float)
    n = len(x)

    def count(mm):
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[: n - m]
        c = 0
        for i in range(len(emb) - 1):
            d = np.max(np.abs(emb[i + 1:] - emb[i]), axis=1)
            c += np.sum(d <= r)
        return c

    b, a = count(m), count(m + 1)
    return -np.log(a / b) if a > 0 and b > 0 else np.inf


def complexity_entropy(x, dx=4):
    """Normalised permutation entropy H and Jensen-Shannon complexity C (Rosso et al., 2007)."""
    import ordpy

    x = np.asarray(x, float)
    h, c = ordpy.complexity_entropy(x[~np.isnan(x)], dx=dx)
    return {"pe_H": float(h), "js_C": float(c)}


# --- cell: forecastability-decomposition ---
def mean_period(component):
    """Mean period of a mode from its zero crossings: n / (crossings / 2)."""
    c = np.asarray(component, float) - np.mean(component)
    crossings = np.sum(np.signbit(c[1:]) != np.signbit(c[:-1]))
    return len(c) / max(crossings / 2, 0.5)


def emd_low_frequency_share(x, threshold_period, max_imf=10):
    """Share of variance in EMD modes (plus residue) slower than threshold_period.

    Empirical mode decomposition (Huang et al., 1998) splits the series into
    intrinsic mode functions from fast to slow. Variances of modes are not
    exactly additive, so shares are computed over the sum of mode variances.
    """
    from PyEMD import EMD

    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    imfs = EMD().emd(x, max_imf=max_imf)
    var = np.array([np.var(m) for m in imfs])
    slow = np.array([mean_period(m) >= threshold_period for m in imfs])
    slow[-1] = True  # the residue (trend) is always slow
    return {"n_modes": len(imfs), "low_freq_share": float(var[slow].sum() / var.sum())}


def vmd_low_frequency_share(x, threshold_period, k=5, alpha=2000):
    """Share of variance in VMD modes whose centre period exceeds threshold_period.

    Variational mode decomposition (Dragomiretskiy & Zosso, 2014) with K modes
    and bandwidth penalty alpha; centre frequencies are in cycles per sample.
    """
    from vmdpy import VMD

    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) % 2:
        x = x[1:]
    u, _, omega = VMD(x - x.mean(), alpha, 0.0, k, 0, 1, 1e-7)
    centre = np.maximum(omega[-1], 1e-12)  # final centre frequencies (cycles / sample)
    var = np.var(u, axis=1)
    slow = (1 / centre) >= threshold_period
    return {"low_freq_share": float(var[slow].sum() / var.sum()),
            "centre_periods": [float(1 / c) for c in centre]}


# --- cell: forecastability-nonlinear ---
def dfa_alpha(x):
    """Detrended fluctuation analysis exponent (Peng et al., 1994) via nolds.

    The slope is fitted by least squares (fit_exp="poly"): nolds' default RANSAC
    fit samples at random, so it gives a slightly different answer on every run.
    """
    import nolds

    x = np.asarray(x, float)
    return float(nolds.dfa(x[~np.isnan(x)], fit_exp="poly"))


def lyapunov_horizon(x, max_points=3000, emb_dim=6, seed=0):
    """Largest Lyapunov exponent (Rosenstein et al., 1993) and tau_f = 1 / lambda.

    Uses the most recent `max_points` points (the method compares every pair of
    delay vectors). On noisy data the estimate is biased, so it is also computed
    on a shuffled copy: only a lambda clearly different from the shuffled one
    says anything about the series. A non-positive lambda gives tau_f = inf.
    The slope is fitted with nolds' default RANSAC, a robust line fit that is not
    pulled by the first jump or the flat tail of the divergence curve. RANSAC
    samples at random, so NumPy's global seed is fixed for the call (and restored
    afterwards) to make the result reproducible.
    """
    import nolds

    x = np.asarray(x, float)
    x = x[~np.isnan(x)][-max_points:]
    shuffled = np.random.default_rng(seed).permutation(x)
    state = np.random.get_state()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(seed)
        lam = float(nolds.lyap_r(x, emb_dim=emb_dim, trajectory_len=20))
        np.random.seed(seed)
        lam_shuf = float(nolds.lyap_r(shuffled, emb_dim=emb_dim, trajectory_len=20))
    np.random.set_state(state)
    return {"lyapunov": lam, "lyapunov_shuffled": lam_shuf,
            "tau_f": (1 / lam) if lam > 0 else np.inf}


def difficulty_rank(table, columns_higher_easier):
    """Average rank across measures where a higher value means easier to forecast."""
    ranks = pd.DataFrame({c: table[c].rank(ascending=False) for c in columns_higher_easier})
    return ranks.mean(axis=1).rank(method="min").astype(int)
