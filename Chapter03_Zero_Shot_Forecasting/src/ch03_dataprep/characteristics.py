"""Global characteristics and summary statistics (Notebook 01).

Each function is a line-by-line port of an R function, so the numbers
match what R reports:

* entropy_mm        -> entropy::entropy(x, method = "MM")
* hurst_exponents   -> pracma::hurstexp()
* keenan_test       -> nonlinearTseries::keenanTest()
* tsay_test         -> nonlinearTseries::tsayTest()
* wo_seasonality    -> seastests::combined_test() (version 0.15.4)
* kpss_short        -> tseries::kpss.test(lshort = TRUE)
* bonferroni_outliers -> car::outlierTest() on y ~ time

The notebook shows every function in full; this file holds the same code so
later notebooks can import it (tests/test_notebook_code_in_sync.py checks
that the two copies are identical).
"""
# --- cell: characteristics-imports ---
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import gammaln
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, bds, kpss


# --- cell: summary-statistics ---
def r_quantile_type1(x, q):
    """Quantile as R's quantile(type = 1): an observed value, x_(ceil(n q))."""
    x = np.sort(np.asarray(x, float))
    j = int(np.ceil(len(x) * q))
    return x[max(j, 1) - 1]


def entropy_mm(x):
    """Miller-Madow entropy exactly as entropy::entropy(x, method = "MM").

    The R function treats the input vector as *counts*: p_i = x_i / sum(x),
    H = -sum_{p_i > 0} p_i log p_i + (m - 1) / (2 sum(x)), m = number of x_i > 0.
    Given a raw series it mostly measures the series' length (NB01). With
    negative values R silently leaves them out of the sum (but not out of the
    total); we do the same so the numbers match, and NB01 flags such series.
    """
    y = np.asarray(x, float)
    y = y[~np.isnan(y)]
    n = y.sum()
    p = y / n
    h = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0))
    m = np.sum(y > 0)
    return float(h + (m - 1) / (2 * n))


def shannon_entropy_hist(x, bins="fd"):
    """Plug-in Shannon entropy (nats) of a histogram of the values."""
    counts, _ = np.histogram(np.asarray(x, float), bins=bins)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log(p)))


def summary_statistics(x):
    """One row of Table 1: summary statistics, n and both entropy measures."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    mean = x.mean()
    sd = x.std(ddof=1)
    return {
        "n": len(x), "min": x.min(), "q1": r_quantile_type1(x, 0.25),
        "median": np.median(x), "mean": mean, "sd": sd,
        "q3": r_quantile_type1(x, 0.75), "max": x.max(),
        "cov_pct": 100 * sd / mean if mean != 0 else np.nan,
        "entropy_mm": entropy_mm(x), "entropy_hist": shannon_entropy_hist(x),
    }


# --- cell: shape ---
def skew_kurt_type3(x):
    """Skewness and excess kurtosis as e1071 type 3 (the package default)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    s = x.std(ddof=1)
    m = x.mean()
    skew = np.mean((x - m) ** 3) / s ** 3
    kurt = np.mean((x - m) ** 4) / s ** 4 - 3
    return skew, kurt


# --- cell: stationarity ---
def kpss_short(x, regression="c"):
    """KPSS test with the short lag of tseries::kpss.test (lshort = TRUE).

    Lag = trunc(4 * (n / 100) ** 0.25). Returns (statistic, p-value). The
    p-value is interpolated from the KPSS table, so it is bounded to
    [0.01, 0.10]; statsmodels warns at the bounds and we silence that.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    nlags = int(np.trunc(4 * (len(x) / 100) ** 0.25))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, *_ = kpss(x, regression=regression, nlags=nlags)
    return stat, p


def stationarity_verdict(x, alpha=0.05):
    """KPSS (level and trend) plus ADF, and a plain-language verdict.

    paper_label is the coarse label of the compact Table 2: "Trend Stationary" when the
    KPSS *level* test does not reject at 5 %, otherwise "Non-stationary".
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    _, p_level = kpss_short(x, "c")
    _, p_trend = kpss_short(x, "ct")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = adfuller(x, autolag="AIC")[1]
    if p_level >= alpha and adf_p < alpha:
        verdict = "level-stationary"
    elif p_trend >= alpha and adf_p < alpha:
        verdict = "trend-stationary"
    elif p_level < alpha and adf_p >= alpha:
        verdict = "difference-stationary (unit root)"
    else:
        verdict = "inconclusive (tests disagree)"
    return {
        "kpss_level_p": p_level, "kpss_trend_p": p_trend, "adf_p": adf_p,
        "verdict": verdict,
        "paper_label": "Trend Stationary" if p_level >= alpha else "Non-stationary",
    }


# --- cell: seasonality ---
def qs_test(y, freq):
    """QS test (seastests::qs): Ljung-Box-type test on the seasonal lags."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    n = len(y)
    yc = y - y.mean()
    c0 = np.sum(yc * yc) / n
    rho = [np.sum(yc[: n - k] * yc[k:]) / n / c0 for k in (freq, 2 * freq)]
    if any(r <= 0 for r in rho):
        rho = [0.0, 0.0]
    qs = n * (n + 2) * (rho[0] ** 2 / (n - freq) + rho[1] ** 2 / (n - 2 * freq))
    return qs, stats.chi2.sf(qs, 2)


def kw_test(y, freq):
    """Kruskal-Wallis test across the seasonal positions (seastests::kw)."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    rank = stats.rankdata(y)
    extra = (freq - len(rank) % freq) % freq
    mat = np.concatenate([np.full(extra, np.nan), rank]).reshape(-1, freq)
    nobs = np.sum(~np.isnan(mat), 0)
    r2n = np.nansum(mat, 0) ** 2 / nobs
    nt = nobs.sum()
    h = 12 / (nt * (nt + 1)) * np.sum(r2n) - 3 * (nt + 1)
    _, cnt = np.unique(rank, return_counts=True)
    if np.any(cnt > 1):
        g = cnt[cnt > 1]
        h = h / (1 - np.sum(g ** 3 - g) / (len(y) ** 3 - len(y)))
    return h, stats.chi2.sf(h, freq - 1)


def _arima_residuals(y, freq):
    """Residuals of the automatic non-seasonal ARIMA used by seastests.

    seastests calls forecast::auto.arima(max.P = 0, max.Q = 0, max.D = 0,
    stepwise = FALSE, max.order = 1 if freq < 8 else 3, allowdrift = freq < 8).
    statsforecast's auto_arima_f is a port of forecast::auto.arima.
    """
    from statsforecast.arima import auto_arima_f

    fit = auto_arima_f(
        np.asarray(y, float), max_p=3, max_q=3, max_P=0, max_Q=0, max_D=0,
        max_order=1 if freq < 8 else 3, seasonal=False, stepwise=False,
        allowdrift=freq < 8, period=1,
    )
    return np.asarray(fit["residuals"], float)


def wo_seasonality(y, freq):
    """Ollech-Webel combined seasonality test, as coded in seastests 0.15.4.

    Seasonal if p(QS on ARIMA residuals) < 0.01 or p(KW on residuals) < 0.001,
    or if p(QS on first differences) < 0.01 and p(KW on residuals) < 0.01.
    (The package help says 0.002 for the second threshold; the code uses
    0.001, and so do we.)
    """
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    p_qs_diff = qs_test(np.diff(y), freq)[1]
    resid = _arima_residuals(y, freq)
    p_qs_res = qs_test(resid, freq)[1]
    p_kw_res = kw_test(resid, freq)[1]
    seasonal = (p_qs_res < 0.01 or p_kw_res < 0.001) or (p_qs_diff < 0.01 and p_kw_res < 0.01)
    return {"freq": freq, "seasonal": bool(seasonal), "p_qs_diff": p_qs_diff,
            "p_qs_resid": p_qs_res, "p_kw_resid": p_kw_res}


def stl_seasonal_strength(y, period, seasonal=13):
    """Seasonal strength F_S = max(0, 1 - Var(R) / Var(S + R)) (Wang, Smith & Hyndman, 2006).

    STL (Cleveland et al., 1990) with a seasonal window of 13 cycles, not robust,
    and the default trend and low-pass windows and speed-up "jumps" of R's stl()
    (every ceil(window / 10)-th point is smoothed and the rest interpolated),
    which keeps period 365 fast.
    """
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    trend = int(np.ceil(1.5 * period / (1 - 1.5 / seasonal)))
    trend += 1 - trend % 2                       # next odd number, as R's nextodd()
    low_pass = period + 2 if period % 2 else period + 1  # smallest odd number > period
    res = STL(y, period=period, seasonal=seasonal, trend=trend, low_pass=low_pass, robust=False,
              seasonal_jump=int(np.ceil(seasonal / 10)), trend_jump=int(np.ceil(trend / 10)),
              low_pass_jump=int(np.ceil(low_pass / 10))).fit()
    return max(0.0, 1 - np.var(res.resid) / np.var(res.seasonal + res.resid))


# --- cell: nonlinearity ---
def _levinson(r, p):
    coefs = np.zeros((p, p))
    variances = np.zeros(p)
    v = r[0]
    phi = np.zeros(0)
    for k in range(1, p + 1):
        kappa = (r[k] - np.dot(phi, r[1:k][::-1])) / v
        phi = np.concatenate([phi - kappa * phi[::-1], [kappa]])
        v = v * (1 - kappa ** 2)
        coefs[k - 1, :k] = phi
        variances[k - 1] = v
    return coefs, variances


def ar_yule_walker_aic(y):
    """AR order by Yule-Walker AIC, as R's ar() default. Returns (p, phi, resid)."""
    y = np.asarray(y, float)
    n = len(y)
    x = y - y.mean()
    pmax = int(min(n - 1, np.floor(10 * np.log10(n))))
    r = np.array([np.sum(x[: n - k] * x[k:]) / n for k in range(pmax + 1)])
    coefs, variances = _levinson(r, pmax)
    aic = n * np.log(np.concatenate([[r[0]], variances])) + 2 * np.arange(pmax + 1) + 2
    p = int(np.argmin(aic))
    phi = coefs[p - 1, :p] if p > 0 else np.zeros(0)
    resid = np.full(n, np.nan)
    for t in range(p, n):
        resid[t] = x[t] - np.dot(phi, x[t - p:t][::-1])
    return p, phi, resid


def _embed(y, order):
    n = len(y)
    return np.column_stack([y[j: n - order + 1 + j] for j in range(order)])


def keenan_test(y):
    """Keenan (1985) one-degree test for non-linearity (nonlinearTseries port)."""
    y = np.asarray(y, float)
    n = len(y)
    p, _, resid = ar_yule_walker_aic(y)
    if p == 0:  # white-noise-like series: test against an AR(1), as a floor
        p, _, resid = _ar_fixed(y, 1)
    fitted = (y - resid)[p:]
    e = resid[p:]
    X = np.column_stack([np.ones(n - p), _embed(y[:-1], p)])
    xi = fitted ** 2 - X @ np.linalg.lstsq(X, fitted ** 2, rcond=None)[0]
    b = np.dot(xi, e) / np.dot(xi, xi)
    fit = b * xi
    rss = np.sum((e - fit) ** 2)
    f0 = np.sum(fit ** 2) / (rss / (len(e) - 1))
    df2 = n - 2 * p - 2
    f = f0 * df2 / (n - p - 1)
    return f, stats.f.sf(f, 1, df2), p


def _ar_fixed(y, p):
    """Yule-Walker AR(p) residuals for a fixed order p (used when AIC picks 0)."""
    y = np.asarray(y, float)
    n = len(y)
    x = y - y.mean()
    r = np.array([np.sum(x[: n - k] * x[k:]) / n for k in range(p + 1)])
    coefs, _ = _levinson(r, p)
    phi = coefs[p - 1, :p]
    resid = np.full(n, np.nan)
    for t in range(p, n):
        resid[t] = x[t] - np.dot(phi, x[t - p:t][::-1])
    return p, phi, resid


def tsay_test(y):
    """Tsay (1986) F test for quadratic non-linearity (nonlinearTseries port)."""
    y = np.asarray(y, float)
    n = len(y)
    order = max(1, ar_yule_walker_aic(y)[0])
    X = _embed(y[:-1], order)
    Y = y[order:]
    lin = sm.OLS(Y, sm.add_constant(X)).fit()
    cross = [y[order - i: n - i] * y[order - j: n - j]
             for i in range(1, order + 1) for j in range(i, order + 1)]
    full = sm.OLS(Y, sm.add_constant(np.column_stack([X] + cross))).fit()
    f, p, _ = full.compare_f_test(lin)
    return f, p, order


def nonlinearity(y, alpha=0.10):
    """Label a series non-linear if Keenan or Tsay rejects linearity at 10 %."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    _, p_keenan, order = keenan_test(y)
    _, p_tsay, _ = tsay_test(y)
    resid = ar_yule_walker_aic(y)[2]
    resid = resid[~np.isnan(resid)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_bds = float(np.atleast_1d(bds(resid[-5000:], max_dim=2)[1])[0])
    return {"ar_order": order, "p_keenan": p_keenan, "p_tsay": p_tsay,
            "p_bds_m2": p_bds,
            "paper_label": "Non-linear" if np.nanmin([p_keenan, p_tsay]) < alpha else "Linear"}


# --- cell: hurst ---
def hurst_exponents(x, d=50):
    """pracma::hurstexp(x, d = 50) port. Returns Hs, Hrs, He, Hal, Ht.

    Table 2 reports Hal, the Anis-Lloyd/Peters corrected empirical Hurst
    exponent: regress log10(R/S - E[R/S] + sqrt(pi n / 2)) on log10(n).
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n_all = len(x)
    if n_all % 2 != 0:
        x = np.append(x, (x[n_all - 2] + x[n_all - 1]) / 2)
        n_all += 1

    def divisors(n, n0=2):
        a = np.arange(n0, n // 2 + 1)
        return a[n % a == 0]

    n0 = min(int(np.floor(0.99 * n_all)), n_all - 1)
    n1, dv = n0, divisors(n0, d)
    for i in range(n0 + 1, n_all + 1):
        dw = divisors(i, d)
        if len(dw) > len(dv):
            n1, dv = i, dw
    x = x[:n1]

    def rs_mean(z, n):
        y = z.reshape(len(z) // n, n).T
        e = y.mean(0)
        s = y.std(0, ddof=1)
        y = np.cumsum(y - e, 0)
        return np.mean((y.max(0) - y.min(0)) / s)

    rs_e = np.array([rs_mean(x, n) for n in dv])
    ers = []
    for n in dv:
        k = np.arange(n - 1, 0, -1) / np.arange(1, n)
        ratio = (n - 0.5) / n * np.sum(np.sqrt(k))
        if n > 340:
            ers.append(ratio / np.sqrt(0.5 * np.pi * n))
        else:
            ers.append(np.exp(gammaln(0.5 * (n - 1)) - gammaln(0.5 * n)) * ratio / np.sqrt(np.pi))
    ers = np.array(ers)
    ers_al = np.sqrt(0.5 * np.pi * dv)
    h_al = np.polyfit(np.log10(dv), np.log10(rs_e - ers + ers_al), 1)[0]
    h_e = np.polyfit(np.log10(dv), np.log10(rs_e), 1)[0]
    h_t = np.polyfit(np.log10(dv), np.log10(ers), 1)[0]
    y = x - x.mean()
    s = np.cumsum(y)
    h_s = np.log((s.max() - s.min()) / x.std(ddof=1)) / np.log(len(x))

    def rs_one(z):
        c = np.cumsum(z - z.mean())
        return (c.max() - c.min()) / z.std(ddof=1)

    def halve(v):
        v = np.asarray(v)
        return np.sort(np.concatenate([v, v[:-1] + ((np.diff(v) + 1) // 2)]))

    xs, ys, cuts = [len(x)], [rs_one(x)], np.array([0, len(x) // 2, len(x)])
    while np.min(np.diff(cuts)) >= 8:
        xl, yl = [], []
        for i in range(1, len(cuts)):
            xl.append(cuts[i] - cuts[i - 1])
            yl.append(rs_one(x[cuts[i - 1]: cuts[i]]))
        xs.append(np.mean(xl))
        ys.append(np.mean(yl))
        cuts = halve(cuts)
    h_rs = np.polyfit(np.log(xs), np.log(ys), 1)[0]
    return {"Hs": h_s, "Hrs": h_rs, "He": h_e, "Hal": h_al, "Ht": h_t}


# --- cell: outliers-and-breaks ---
def bonferroni_outliers(y, alpha=0.05):
    """Number of Bonferroni-significant studentised residuals of y = a + b t + e.

    As car::outlierTest(lm(y ~ t)): externally studentised residuals
        r_i = e_i / (s_(i) sqrt(1 - h_ii)),
        s_(i)^2 = ((n - k) s^2 - e_i^2 / (1 - h_ii)) / (n - k - 1),
    each compared with a t distribution on n - k - 1 degrees of freedom, and the
    two-sided p-value multiplied by n (Bonferroni). Closed form, so it is fast
    for 10,000 points. (car prints at most 10 of them; we count all.)
    """
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    n = len(y)
    X = np.column_stack([np.ones(n), np.arange(1, n + 1)])
    k = X.shape[1]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)
    s2 = e @ e / (n - k)
    s2_i = ((n - k) * s2 - e ** 2 / (1 - h)) / (n - k - 1)
    r = e / np.sqrt(s2_i * (1 - h))
    p_bonf = np.minimum(1.0, n * 2 * stats.t.sf(np.abs(r), n - k - 1))
    return int((p_bonf < alpha).sum())


def ols_cusum(y):
    """OLS-CUSUM test of a stable mean and trend (Ploberger & Kraemer, 1992)."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    t = np.arange(len(y))
    res = sm.OLS(y, sm.add_constant(t)).fit()
    stat, p, _ = breaks_cusumolsresid(res.resid, ddof=2)
    return stat, p


def pelt_breaks(y, index, penalty_scale=5.0, min_size=24):
    """Dates where the mean or the variance of the changes shifts, by PELT (ruptures), for the plots.

    PELT (Killick, Fearnhead & Eckley, 2012) minimises the sum of segment costs
    plus a penalty per break. Cost: Gaussian negative log-likelihood of each
    segment of the changes d_t = y_t - y_{t-1} (ruptures model "normal", so a
    shift in level *or* volatility of the changes counts); penalty 5 log(n);
    segments of at least min_size points. Changes rather than levels, because a
    random walk has no stable level to shift from and level-based PELT would
    cut it into many spurious pieces.
    """
    import ruptures as rpt

    d = np.diff(np.asarray(y, float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algo = rpt.Pelt(model="normal", min_size=min_size).fit(d.reshape(-1, 1))
        bkps = algo.predict(pen=penalty_scale * np.log(len(d)))
    return [index[b + 1] for b in bkps[:-1]]


# --- cell: global-table ---
def global_characteristics(x, seasonal_periods):
    """One row of Table 2 for a series: compact labels plus the detail columns."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    skew, kurt = skew_kurt_type3(x)
    st = stationarity_verdict(x)
    nl = nonlinearity(x)
    h = hurst_exponents(x)
    seas = {p: wo_seasonality(x, p) for p in seasonal_periods}
    strength = {}
    for p in seasonal_periods:
        try:
            strength[p] = stl_seasonal_strength(x, p)
        except Exception:  # series too short for this period
            strength[p] = np.nan
    cusum_stat, cusum_p = ols_cusum(x)
    row = {
        "skewness": skew, "kurtosis": kurt,
        "nonlinearity": nl["paper_label"], "p_keenan": nl["p_keenan"],
        "p_tsay": nl["p_tsay"], "p_bds": nl["p_bds_m2"],
        "stationarity_paper": st["paper_label"], "stationarity": st["verdict"],
        "p_kpss_level": st["kpss_level_p"], "p_kpss_trend": st["kpss_trend_p"],
        "p_adf": st["adf_p"], "hurst_Hal": h["Hal"],
        "outliers_bonf": bonferroni_outliers(x),
        "cusum_stat": cusum_stat, "p_cusum": cusum_p,
    }
    for p in seasonal_periods:
        row[f"seasonal_{p}"] = "Seasonal" if seas[p]["seasonal"] else "Non-seasonal"
        row[f"stl_strength_{p}"] = strength[p]
    return row
