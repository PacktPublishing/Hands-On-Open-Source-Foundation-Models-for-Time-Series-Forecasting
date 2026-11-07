"""Granger causality (linear, non-linear, by time scale), SURD and rolling correlation (Notebook 02).

* Linear Granger causality: bivariate VAR, lag order by the AIC of the target's own
  autoregression, Wald F test of the driver's lags with HC3 standard errors (Granger, 1969; MacKinnon & White, 1985).
* Non-linear Granger causality: the Diks & Panchenko (2006) test, applied to the
  residuals of the linear VAR (as Hiemstra & Jones, 1994, proposed), so that it
  responds only to dependence the linear test cannot see. Lags lx = ly = 1,
  bandwidth eps = 8 n^(-2/7) on standardised data, autocorrelation-robust
  variance with Bartlett weights.
* Granger causality by time scale: the linear test applied to causal Haar MODWT
  bands of both series (scale-by-scale analysis as in Ramsey & Lampart, 1998).
  Band-pass filtering breaks the textbook F distribution (Barnett & Seth, 2011),
  so p-values come from phase-randomised surrogates of the driver (Theiler et
  al., 1992), which keep its spectrum and destroy any link with the target.
* SURD (Martinez-Sanchez, Arranz & Lozano-Duran, 2024): the information that the
  present of the sources carries about the future of the target, split into
  redundant, unique and synergistic parts plus the causality leak. Computed with
  the authors' package `surd` on equal-count bins; a surrogate baseline gives the
  share expected from estimation bias alone.
* Rolling Spearman correlation with an approximate 95 % band under independence.
"""
# --- cell: causality-granger ---
import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS


def lag_matrix(v, p):
    """Columns v[t-1], ..., v[t-p] for t = p, ..., n - 1."""
    v = np.asarray(v, float)
    return np.column_stack([v[p - k: len(v) - k] for k in range(1, p + 1)])


def select_lag(y, max_lag):
    """Lag order p of the target's own autoregression, by AIC on a common sample.

    The choice looks only at y, never at x: choosing p from a model that already
    contains x lets the data pick lags where x happens to fit, and the test then
    rejects too often (about 7 % instead of 5 % in simulations).
    """
    y = np.asarray(y, float)
    target = y[max_lag:]
    best, best_aic = 1, np.inf
    for p in range(1, max_lag + 1):
        design = np.column_stack([np.ones(len(target)), lag_matrix(y, max_lag)[:, :p]])
        rss = np.sum((target - design @ np.linalg.lstsq(design, target, rcond=None)[0]) ** 2)
        aic = len(target) * np.log(rss / len(target)) + 2 * (p + 1)
        if aic < best_aic:
            best, best_aic = p, aic
    return best


def granger_linear(y, x, max_lag, lag=None):
    """Does the past of x improve a linear forecast of y beyond the past of y?

    Model y_t = c + sum_i a_i y_{t-i} + sum_i b_i x_{t-i} + e_t, H0: b_1 = ... = b_p = 0,
    Wald F test with heteroskedasticity-robust (HC3) covariance; p from the AIC of
    y's own autoregression. Also returns the residuals of both VAR equations, the
    input of the non-linear test.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    p = lag or select_lag(y, max_lag)
    design = np.column_stack([np.ones(len(y) - p), lag_matrix(y, p), lag_matrix(x, p)])
    fit_y = OLS(y[p:], design).fit(cov_type="HC3")
    restriction = np.zeros((p, design.shape[1]))
    restriction[:, 1 + p:] = np.eye(p)
    wald = fit_y.wald_test(restriction, use_f=True, scalar=True)
    fit_x = OLS(x[p:], design).fit()
    return {"lag": p, "F": float(wald.statistic), "p_value": float(wald.pvalue),
            "resid_y": fit_y.resid, "resid_x": fit_x.resid}


def _standardise(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=1)


def diks_panchenko(x, y, eps=None, chunk=1500):
    """Diks & Panchenko (2006) test of H0: x does not Granger-cause y (lx = ly = 1).

    With X = x_{t-1}, Y = y_{t-1}, Z = y_t and local densities estimated by counting
    neighbours within eps (sup norm) on standardised data,
        T_n = mean_i [ f_XYZ(i) f_Y(i) - f_XY(i) f_YZ(i) ],
    which is zero under H0. sqrt(n) T_n / S_n is asymptotically N(0, 1); S_n^2 is 9 times
    the long-run variance of the U-statistic projection, with Bartlett weights and
    floor(n^(1/4)) lags. One-sided: large values reject.
    """
    x, y = _standardise(x), _standardise(y)
    X, Y, Z = x[:-1], y[:-1], y[1:]
    n = len(Z)
    eps = 8.0 * n ** (-2 / 7) if eps is None else eps

    def blocks():
        for start in range(0, n, chunk):
            rows = slice(start, min(start + chunk, n))
            ax = np.abs(X[rows, None] - X[None, :]) <= eps
            ay = np.abs(Y[rows, None] - Y[None, :]) <= eps
            az = np.abs(Z[rows, None] - Z[None, :]) <= eps
            ay[np.arange(ay.shape[0]), np.arange(rows.start, rows.stop)] = False   # j != i
            axy, ayz = ax & ay, ay & az
            yield rows, ay, axy, ayz, axy & az

    counts = {k: np.zeros(n) for k in ("y", "xy", "yz", "xyz")}
    for rows, ay, axy, ayz, axyz in blocks():
        for k, a in zip(("y", "xy", "yz", "xyz"), (ay, axy, ayz, axyz)):
            counts[k][rows] = a.sum(axis=1)
    cy, cxy, cyz, cxyz = (counts[k] for k in ("y", "xy", "yz", "xyz"))
    cross = np.zeros(n)
    for rows, ay, axy, ayz, axyz in blocks():
        cross[rows] = ay @ cxyz + axyz @ cy - ayz @ cxy - axy @ cyz
    scale = 2.0 / (2 * eps) ** 4 / (6.0 * (n - 1) ** 2)
    h = scale * (cxyz * cy - cxy * cyz + cross)
    t_n = h.mean()
    hc = h - t_n
    k_max = int(np.floor(n ** 0.25))
    weights = np.concatenate([[1.0], 2.0 * (1.0 - np.arange(1, k_max) / k_max)])
    autocov = np.array([np.dot(hc[k:], hc[:n - k]) / (n - k) for k in range(k_max)])
    variance = 9.0 * np.dot(weights, autocov)
    t_stat = t_n * np.sqrt(n) / np.sqrt(variance)
    return {"eps": eps, "T": t_n, "t_stat": float(t_stat), "p_value": float(stats.norm.sf(t_stat))}


# --- cell: causality-wavelet ---
from ch03_dataprep.features import causal_haar_modwt


def phase_randomise(v, rng):
    """Surrogate with the same mean and power spectrum as v and random Fourier phases."""
    v = np.asarray(v, float)
    spec = np.fft.rfft(v - v.mean())
    phases = rng.uniform(0, 2 * np.pi, len(spec))
    phases[0] = 0.0
    if len(v) % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=len(v)) + v.mean()


def scale_bands(v, unit):
    """Causal Haar MODWT bands W1+W2, W3+W4, W5+W6 (periods 2-8, 8-32, 32-128 steps)."""
    c = causal_haar_modwt(np.asarray(v, float), levels=6)
    return {f"2-8 {unit}": c["W1"] + c["W2"], f"8-32 {unit}": c["W3"] + c["W4"],
            f"32-128 {unit}": c["W5"] + c["W6"]}


def _f_stat(y, own, other, p):
    """Classical Granger F statistic, given the restricted design `own`."""
    full = np.column_stack([own, lag_matrix(other, p)])
    target = y[p:]
    rss_r = np.sum((target - own @ np.linalg.lstsq(own, target, rcond=None)[0]) ** 2)
    rss_u = np.sum((target - full @ np.linalg.lstsq(full, target, rcond=None)[0]) ** 2)
    dof = len(target) - full.shape[1]
    return ((rss_r - rss_u) / p) / (rss_u / dof)


def wavelet_granger(y, x, unit, max_lag, n_surrogates=199, seed=0):
    """Granger F test band by band, with a phase-randomised surrogate p-value per band."""
    burn = 2 ** 6 - 1                                  # first 63 values depend on the start
    yb = {k: v[burn:] for k, v in scale_bands(y, unit).items()}
    xb = {k: v[burn:] for k, v in scale_bands(x, unit).items()}
    rng = np.random.default_rng(seed)
    setup = {}
    for band in yb:
        p = select_lag(yb[band], max_lag)
        own = np.column_stack([np.ones(len(yb[band]) - p), lag_matrix(yb[band], p)])
        setup[band] = (p, own, _f_stat(yb[band], own, xb[band], p))
    null = {band: [] for band in yb}
    for _ in range(n_surrogates):
        sb = scale_bands(phase_randomise(x, rng), unit)
        for band, (p, own, _) in setup.items():
            null[band].append(_f_stat(yb[band], own, sb[band][burn:], p))
    out = []
    for band, (p, own, f_obs) in setup.items():
        exceed = int(np.sum(np.asarray(null[band]) >= f_obs))
        out.append({"band": band, "lag": p, "F": f_obs, "null_F_95": float(np.quantile(null[band], 0.95)),
                    "p_value": (1 + exceed) / (1 + n_surrogates)})
    return out


# --- cell: causality-surd ---
import surd as surd_pkg


def equal_count_bins(v, bins):
    """Bin index 0..bins-1 with (almost) the same number of values in every bin."""
    ranks = stats.rankdata(v, method="ordinal") - 1
    return (ranks * bins // len(ranks)).astype(int)


def surd_shares(target, sources, lag=1, bins=4):
    """SURD of target[t + lag] given sources[t], as shares of the target's entropy.

    `sources` is a dict name -> series (same length as target). Returns a dict with
    'unique <name>', 'redundant <a+b...>', 'synergy <a+b...>' shares and 'leak';
    together they add up to 1.
    """
    names = list(sources)
    cols = [equal_count_bins(np.asarray(target, float)[lag:], bins)]
    cols += [equal_count_bins(np.asarray(sources[k], float)[:-lag], bins) for k in names]
    edges = [np.arange(bins + 1) - 0.5] * len(cols)
    hist, _ = np.histogramdd(np.column_stack(cols), bins=edges)
    p_target = hist.sum(axis=tuple(range(1, hist.ndim))) / hist.sum()
    h_target = -np.sum(p_target[p_target > 0] * np.log2(p_target[p_target > 0]))
    i_r, i_s, _, leak = surd_pkg.surd(hist)
    out = {}
    for key, val in i_r.items():
        label = "unique " + names[key[0] - 1] if len(key) == 1 else "redundant " + "+".join(names[k - 1] for k in key)
        out[label] = float(val / h_target)
    for key, val in i_s.items():
        out["synergy " + "+".join(names[k - 1] for k in key)] = float(val / h_target)
    out["leak"] = float(leak)
    return out


def surd_with_null(target, sources, drivers, lag=1, bins=4, n_surrogates=100, seed=0):
    """SURD shares plus a null: the same shares with every driver phase-randomised.

    Returns (shares, null) where null is a DataFrame of surrogate shares (one row per
    surrogate). Terms that involve a driver should exceed their null 95th percentile.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    shares = surd_shares(target, sources, lag, bins)
    rows = []
    for _ in range(n_surrogates):
        fake = {k: (phase_randomise(v, rng) if k in drivers else v) for k, v in sources.items()}
        rows.append(surd_shares(target, fake, lag, bins))
    return shares, pd.DataFrame(rows)


# --- cell: causality-rolling ---
def rolling_spearman(y, x, window, lag=0):
    """Spearman correlation of y[t] and x[t - lag] in trailing windows of `window` points.

    Returns (correlation series indexed by window end, half-width of an approximate 95 %
    band under independence, 1.96 * sqrt(1.06 / (window - 3)); Fieller et al., 1957).
    """
    import pandas as pd
    from numpy.lib.stride_tricks import sliding_window_view
    pair = pd.concat([y.rename("y"), x.shift(lag).rename("x")], axis=1).dropna()
    wy = sliding_window_view(pair["y"].to_numpy(), window)
    wx = sliding_window_view(pair["x"].to_numpy(), window)
    ry = wy.argsort(axis=1).argsort(axis=1).astype(float)
    rx = wx.argsort(axis=1).argsort(axis=1).astype(float)
    ry -= ry.mean(axis=1, keepdims=True)
    rx -= rx.mean(axis=1, keepdims=True)
    rho = (ry * rx).sum(axis=1) / np.sqrt((ry ** 2).sum(axis=1) * (rx ** 2).sum(axis=1))
    return pd.Series(rho, index=pair.index[window - 1:]), 1.96 * np.sqrt(1.06 / (window - 3))
