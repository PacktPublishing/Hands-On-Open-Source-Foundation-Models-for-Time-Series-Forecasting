"""Fixed-origin and rolling-origin splits by horizon (Notebook 04; read by every later notebook)."""
# --- cell: splits ---
import pandas as pd


def windows_for(grain, horizon, rolling_windows):
    """Number of rolling windows R for a grain and horizon (plan section 5)."""
    return int(rolling_windows[grain][horizon])


def make_splits(index, dataset, grain, horizons, rolling_windows):
    """One row per (horizon, window): origin T_r = N - r H, context 1..T_r, test T_r+1..T_r+H.

    Window r = 1 is the primary (fixed-origin) split; r = 2..R move the origin
    back by whole horizons, so the R test windows of a horizon never overlap.
    Positions are 0-based row numbers; dates are the index values at them.
    """
    idx = pd.DatetimeIndex(index)
    n = len(idx)
    rows = []
    for h in horizons:
        for r in range(1, windows_for(grain, h, rolling_windows) + 1):
            origin = n - r * h            # number of context rows, T_r
            rows.append({
                "dataset": dataset, "grain": grain, "horizon": h, "window": r,
                "primary": r == 1, "origin_date": idx[origin - 1],
                "context_start": idx[0], "context_end": idx[origin - 1],
                "test_start": idx[origin], "test_end": idx[origin + h - 1],
                "n_context": origin, "n_test": h,
            })
    return pd.DataFrame(rows)


def insample_end(manifest):
    """Last date before the earliest test window of any split: the end of the
    sample that exploratory analysis (NB02 wavelet coherence) may use."""
    earliest = manifest["test_start"].min()
    ends = manifest.loc[manifest["test_start"] == earliest, "context_end"]
    return ends.iloc[0]


def split_frame(frame, row, past_only=(), known_future=()):
    """Model inputs for one manifest row.

    Returns (context, future, test):
      context: target and every covariate up to the origin (rows 1..T);
      future:  known-future covariates over the test window (rows T+1..T+H);
               past-only covariates are never included here;
      test:    the target over the test window (used only for scoring).
    """
    t = int(row["n_context"])
    h = int(row["n_test"])
    context = frame.iloc[:t]
    future = frame.iloc[t:t + h][list(known_future)]
    test = frame.iloc[t:t + h].drop(columns=list(past_only) + list(known_future), errors="ignore")
    assert_no_leakage(context, future, test, row, past_only)
    return context, future, test


def assert_no_leakage(context, future, test, row, past_only=()):
    """Leakage checks: the context ends at the origin, the test window starts the
    next row, and no past-only covariate appears in the future block."""
    assert context.index.max() == row["origin_date"], "context does not end at the origin"
    assert test.index.min() == row["test_start"], "test window does not start after the origin"
    assert context.index.max() < test.index.min(), "context overlaps the test window"
    assert not set(past_only) & set(future.columns), "past-only covariate in the future block"
    assert len(test) == row["n_test"], "test window has the wrong length"


def checkpoint_overlap(manifest, checkpoints):
    """For each split, the models whose weights were last committed on or after
    the first test date.

    A checkpoint cannot contain data from after its weights were uploaded, so
    a test window that starts after that date is guaranteed unseen by it.
    Windows that start earlier *could* be in its pre-training corpus (only if
    the corpus contains this series); they are flagged, not excluded.
    """
    dates = pd.to_datetime(checkpoints.set_index("model")["weights_last_commit_utc"])
    out = manifest.copy()
    out["models_weights_after_test_start"] = [
        ", ".join(sorted(dates.index[dates >= ts])) for ts in out["test_start"]]
    out["n_models_flagged"] = [int((dates >= ts).sum()) for ts in out["test_start"]]
    return out
