"""Metrics, tables and robustness tools of Notebooks 11 and 12, checked on hand-computed examples."""
import numpy as np
import pandas as pd
import pytest

from ch03_zeroshot.metrics import aggregate, naive_forecast, point_metrics
from ch03_zeroshot.robustness import dominance, elementary_scores, mcb
from ch03_zeroshot.tables import rank_marks


def test_naive_and_seasonal_naive():
    h = np.arange(10.0)
    assert naive_forecast(h, 3, 1).tolist() == [9, 9, 9]
    assert naive_forecast(h, 5, 4).tolist() == [6, 7, 8, 9, 6]


def test_point_metrics_by_hand():
    history = np.array([1.0, 2.0, 4.0, 7.0])           # in-sample naive errors 1, 2, 3 -> scale 2
    y, yhat = np.array([8.0, 10.0]), np.array([9.0, 7.0])
    m = point_metrics(y, yhat, history, lag=1)
    assert m["MAE"] == pytest.approx(2.0)
    assert m["RMSE"] == pytest.approx(np.sqrt(5.0))
    assert m["ME"] == pytest.approx(1.0)                # (-1 + 3) / 2
    assert m["MASE"] == pytest.approx(1.0)
    assert m["RMSSE"] == pytest.approx(np.sqrt(5.0 / (14 / 3)))
    assert m["MDAPE"] == pytest.approx(100 * np.median([1 / 8, 3 / 10]))
    assert m["sMAPE"] == pytest.approx(100 * np.mean([1 / 8.5, 3 / 8.5]))
    assert m["TU1"] == pytest.approx(np.sqrt(5) / (np.sqrt((64 + 100) / 2) + np.sqrt((81 + 49) / 2)))
    # naive forecast 7, 7: errors 1, 3 -> relative errors 1 and 1
    assert m["MDRAE"] == pytest.approx(1.0)
    assert m["FVA"] == pytest.approx(0.0)


def test_mdrae_skips_steps_where_naive_is_exact():
    m = point_metrics(np.array([7.0, 9.0]), np.array([8.0, 8.0]), np.array([5.0, 7.0]), lag=1)
    assert m["MDRAE"] == pytest.approx(0.5)             # first step: naive exact, skipped


def test_aggregate_pools_gmrae_geometrically():
    s = pd.DataFrame({"dataset": "D", "series": "x", "horizon": 3, "model": "m", "window": [1, 2], "MAE": [1.0, 3.0],
                      "log_rae_sum": [np.log(2) * 2, np.log(0.5) * 2], "log_rae_n": [2, 2]})
    a = aggregate(s)
    assert a["MAE"].item() == pytest.approx(2.0)
    assert a["GMRAE"].item() == pytest.approx(1.0)


def test_rank_marks_best_and_second():
    idx = pd.MultiIndex.from_tuples([("x", 3, "MASE"), ("x", 3, "ME (bias)")], names=["Variable", "Horizon", "Metric"])
    t = pd.DataFrame([[0.9, 0.7, 0.8], [-0.1, 0.5, 0.2]], index=idx, columns=["A", "B", "C"])
    marks = rank_marks(t)
    assert marks.loc[("x", 3, "MASE")].tolist() == ["", "best", "second"]
    assert marks.loc[("x", 3, "ME (bias)")].tolist() == ["best", "", "second"]   # closest to zero wins


def test_elementary_scores_integrate_to_mae_and_mse():
    rng = np.random.default_rng(0)
    y, x = rng.normal(size=400), rng.normal(size=400)
    th = np.linspace(-6, 6, 24001)
    med = np.trapezoid(elementary_scores(y, x, th, "median"), th)
    mean = np.trapezoid(elementary_scores(y, x, th, "mean"), th)
    assert med == pytest.approx(np.mean(np.abs(y - x)) / 2, rel=1e-3)
    assert mean == pytest.approx(np.mean((y - x) ** 2) / 4, rel=1e-3)


def test_dominance_of_a_better_forecast():
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    th = np.linspace(-4, 4, 81)
    curves = {"good": elementary_scores(y, y, th), "bad": elementary_scores(y, y + 1, th)}
    d = dominance(th, curves)
    assert d.loc["good", "bad"] and not d.loc["bad", "good"]


def test_mcb_critical_difference():
    rng = np.random.default_rng(2)
    rows = [{"dataset": "D", "series": "x", "horizon": 1, "window": t, "model": m, "MASE": v + rng.normal(0, 0.01)}
            for t in range(50) for m, v in (("a", 1.0), ("b", 2.0), ("c", 3.0))]
    res = mcb(pd.DataFrame(rows))
    # Nemenyi CD for 3 models, alpha 0.05: q = 2.343 (Demsar 2006, Table 5)
    assert res.attrs["cd"] == pytest.approx(2.343 * np.sqrt(3 * 4 / (6 * 50)), rel=1e-3)
    assert res.index.tolist() == ["a", "b", "c"] and res["worse_than_best"].tolist() == [False, True, True]
