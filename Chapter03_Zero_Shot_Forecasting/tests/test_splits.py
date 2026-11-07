"""Splits: origins, sizes, non-overlap and leakage checks."""
import numpy as np
import pandas as pd
import pytest

from ch03_dataprep import config, splits


@pytest.fixture
def daily():
    idx = pd.date_range(end="2026-09-30", periods=config.N_DAILY, freq="D")
    return idx, splits.make_splits(idx, "D1", "daily", config.HORIZONS["daily"], config.ROLLING_WINDOWS)


def test_daily_counts_and_origins(daily):
    idx, m = daily
    assert len(m) == 25
    for _, r in m.iterrows():
        assert r["n_context"] == len(idx) - r["window"] * r["horizon"]
        assert (r["test_end"] - r["test_start"]).days == r["horizon"] - 1
    assert (m.query("window == 1")["test_end"] == idx[-1]).all()


def test_rolling_windows_do_not_overlap(daily):
    _, m = daily
    for h, g in m.groupby("horizon"):
        g = g.sort_values("test_start")
        assert (g["test_start"].iloc[1:].to_numpy() > g["test_end"].iloc[:-1].to_numpy()).all()


def test_monthly_counts():
    idx = pd.date_range("1982-01-01", "2026-09-01", freq="MS")
    m = splits.make_splits(idx, "D3", "monthly", config.HORIZONS["monthly"], config.ROLLING_WINDOWS)
    assert m.groupby("horizon").size().to_dict() == {3: 3, 6: 3, 9: 3, 12: 3, 24: 1}
    assert splits.insample_end(m) == pd.Timestamp("2023-09-01")


def test_split_frame_blocks_past_only_covariates(daily):
    idx, m = daily
    frame = pd.DataFrame({"temp": np.arange(len(idx), dtype=float), "nao": 0.0, "daylength_h": 1.0}, index=idx)
    ctx, fut, test = splits.split_frame(frame, m.iloc[0], past_only=["nao"], known_future=["daylength_h"])
    assert list(fut.columns) == ["daylength_h"]
    assert list(test.columns) == ["temp"]
    assert ctx.index[-1] < test.index[0]
    with pytest.raises(AssertionError):
        splits.assert_no_leakage(ctx, frame.iloc[len(ctx):len(ctx) + 7][["nao"]], test, m.iloc[0], past_only=["nao"])


def test_checkpoint_overlap_flags_later_weights():
    idx = pd.date_range("2020-01-01", periods=100, freq="MS")
    m = splits.make_splits(idx, "D3", "monthly", [3], {"monthly": {3: 1}})
    ck = pd.DataFrame({"model": ["old", "new"], "weights_last_commit_utc": ["2020-01-01", "2030-01-01"]})
    out = splits.checkpoint_overlap(m, ck)
    assert out["models_weights_after_test_start"].iloc[0] == "new"
