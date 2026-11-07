"""Robustness checks (Notebook 12): Multiple Comparisons with the Best (MCB) and Murphy diagrams.

MCB (Koning, Franses, Hibon and Stekler, 2005): rank the M models on each of N forecasting tasks,
average the ranks, and draw an interval of half-width CD/2 around each mean rank, with
CD = q_alpha * sqrt(M (M + 1) / (6 N)) and q_alpha the studentised range quantile divided by sqrt(2)
(the Nemenyi critical difference; Demsar, 2006). A model whose interval does not overlap the best
model's interval is significantly worse than the best.

Murphy diagrams (Ehm, Gneiting, Jordan and Krueger, 2016): every consistent scoring function for a
mean or a median is a mixture of 'elementary' scores S_theta. Plotting the average elementary score
against theta shows whether one forecast is better for *every* consistent loss (its curve is lower
everywhere) or only for some.
"""
# --- cell: robustness ---
import numpy as np
import pandas as pd


def mcb(scores, metric="MASE", task=("dataset", "series", "horizon", "window"), alpha=0.05):
    """Mean rank and MCB interval of every model; ``scores`` has one row per model and task."""
    from scipy.stats import studentized_range

    wide = scores.pivot_table(index=list(task), columns="model", values=metric).dropna()
    ranks = wide.rank(axis=1)
    n, m = ranks.shape
    q = studentized_range.ppf(1 - alpha, m, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(m * (m + 1) / (6 * n))
    out = pd.DataFrame({"mean_rank": ranks.mean()}).sort_values("mean_rank")
    out["lower"], out["upper"] = out["mean_rank"] - cd / 2, out["mean_rank"] + cd / 2
    best = out.iloc[0]
    out["worse_than_best"] = out["lower"] > best["upper"]
    out.attrs.update(cd=cd, n_tasks=n, n_models=m, alpha=alpha, metric=metric)
    return out


def plot_mcb(res, ax=None, names=None, title=None):
    """MCB plot: mean rank with interval; the best model's interval is shaded."""
    import matplotlib.pyplot as plt

    names = names or {}
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 0.28 * len(res) + 1.2))
    y = np.arange(len(res))[::-1]
    best = res.iloc[0]
    ax.axvspan(best["lower"], best["upper"], color="#d9d9d9", zorder=0)
    for yi, (k, r) in zip(y, res.iterrows()):
        c = "#b2182b" if r["worse_than_best"] else "#2166ac"
        ax.plot([r["lower"], r["upper"]], [yi, yi], color=c, lw=1.6)
        ax.plot(r["mean_rank"], yi, "o", color=c, ms=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{names.get(k, k)} ({r['mean_rank']:.1f})" for k, r in res.iterrows()], fontsize=8)
    ax.set_xlabel(f"mean rank on {res.attrs['metric']} (lower is better)")
    ax.set_title(title or f"MCB: {res.attrs['n_tasks']} tasks, {res.attrs['n_models']} models, "
                 f"CD = {res.attrs['cd']:.2f}", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    return ax


def elementary_scores(y, x, thetas, functional="median"):
    """Average elementary score at each theta (Ehm et al., 2016, Sec. 2).

    median (quantile level 1/2):  S_theta = 1/2 * 1{min(x, y) <= theta < max(x, y)}; integrates to MAE / 2.
    mean (expectile level 1/2):   S_theta = 1/2 * |y - theta| * 1{min(x, y) <= theta < max(x, y)};
                                  integrates to MSE / 4.
    """
    y, x = np.asarray(y, float)[:, None], np.asarray(x, float)[:, None]
    th = np.asarray(thetas, float)[None, :]
    between = (np.minimum(x, y) <= th) & (th < np.maximum(x, y))
    s = 0.5 * between if functional == "median" else 0.5 * np.abs(y - th) * between
    return s.mean(axis=0)


def murphy(forecasts, models, functional="median", n_theta=201, standardise=True):
    """Murphy curves of several models on the same forecast cases.

    forecasts: long format (model, series, horizon, window, step, y_true, y_pred). With several series of
    different scales, set standardise=True: each series is centred and scaled with its own test values.
    Returns (thetas, {model: curve}).
    """
    d = forecasts[forecasts["model"].isin(models)].copy()
    if standardise:
        g = d.groupby("series")["y_true"]
        d["y_pred"] = (d["y_pred"] - g.transform("mean")) / g.transform("std")
        d["y_true"] = (d["y_true"] - g.transform("mean")) / g.transform("std")
    wide = d.pivot_table(index=["series", "horizon", "window", "step"], columns="model", values="y_pred")
    y = d.drop_duplicates(["series", "horizon", "window", "step"]).set_index(
        ["series", "horizon", "window", "step"])["y_true"].reindex(wide.index)
    wide = wide.dropna()
    y = y.loc[wide.index]
    lo = np.nanmin([y.min(), wide.min().min()])
    hi = np.nanmax([y.max(), wide.max().max()])
    thetas = np.linspace(lo, hi, n_theta)
    return thetas, {m: elementary_scores(y, wide[m], thetas, functional) for m in models if m in wide}


def murphy_difference(forecasts, a, b, functional="median", n_theta=201, n_boot=300, block=None, seed=0):
    """Score difference a - b at each theta with a pointwise 95% band from a moving-block bootstrap
    over forecast cases (negative = a is better at that theta)."""
    thetas, curves = murphy(forecasts, [a, b], functional, n_theta)
    d = forecasts[forecasts["model"].isin([a, b])].copy()
    g = d.groupby("series")["y_true"]
    for c in ("y_pred", "y_true"):
        d[c] = (d[c] - g.transform("mean")) / g.transform("std")
    wide = d.pivot_table(index=["series", "horizon", "window", "step"], columns="model", values="y_pred").dropna()
    y = d.drop_duplicates(["series", "horizon", "window", "step"]).set_index(
        ["series", "horizon", "window", "step"])["y_true"].reindex(wide.index).to_numpy()
    th = thetas[None, :]

    def per_case(x):
        x = x[:, None]
        between = (np.minimum(x, y[:, None]) <= th) & (th < np.maximum(x, y[:, None]))
        return 0.5 * between if functional == "median" else 0.5 * np.abs(y[:, None] - th) * between

    diff = per_case(wide[a].to_numpy()) - per_case(wide[b].to_numpy())
    n = len(diff)
    block = block or max(1, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, int(np.ceil(n / block)))])[:n]
        boots.append(diff[idx].mean(axis=0))
    boots = np.array(boots)
    return thetas, diff.mean(axis=0), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def plot_murphy(thetas, curves, ax=None, names=None, title="", functional="median"):
    import matplotlib.pyplot as plt

    names = names or {}
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.6))
    for m, c in curves.items():
        ax.plot(thetas, c, lw=1.4, label=names.get(m, m))
    ax.set_xlabel("threshold θ (standardised units)")
    ax.set_ylabel(f"mean elementary score ({functional})")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    return ax


def dominance_share(thetas, curves, tol=1e-12):
    """For every pair (a, b): % of the thresholds at which the two curves differ where a scores lower than b.
    100 means a dominates b; values near 50 mean the ranking depends on the threshold."""
    keys = list(curves)
    out = pd.DataFrame(np.nan, index=keys, columns=keys)
    for a in keys:
        for b in keys:
            differ = np.abs(curves[a] - curves[b]) > tol
            if a != b and differ.any():
                out.loc[a, b] = 100 * np.mean(curves[a][differ] < curves[b][differ])
    return out


def dominance(thetas, curves, tol=1e-12):
    """For every pair (a, b): True when a's curve is never above b's (a dominates b for every consistent
    loss of this functional; Ehm et al., 2016, Corollary 1)."""
    keys = list(curves)
    return pd.DataFrame([[bool(np.all(curves[a] <= curves[b] + tol)) if a != b else np.nan for b in keys]
                         for a in keys], index=keys, columns=keys)
