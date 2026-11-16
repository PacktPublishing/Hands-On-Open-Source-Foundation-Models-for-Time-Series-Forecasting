"""A deterministic, synthetic feature-alignment example for Chapter 1.

Run from any working directory: python feature_example.py.
The integer ramp makes a one-hour target shift directly inspectable; these
values are a teaching illustration, not ETTh1 measurements or forecasts.
"""
import numpy as np
import pandas as pd


def lagged_training_rows(y: pd.Series) -> pd.DataFrame:
    """Index rows by target time; every feature precedes that target."""
    return pd.DataFrame({
        "target": y,
        "lag_1": y.shift(1),
        "lag_24": y.shift(24),
        "mean_24": y.shift(1).rolling(24).mean(),
        "hour": y.index.hour,
    }).dropna()


if __name__ == "__main__":
    y = pd.Series(np.arange(72, dtype=float),
                  index=pd.date_range("2020-01-01", periods=72, freq="h"))
    rows = lagged_training_rows(y)
    first = rows.iloc[0]
    assert first["target"] == 24 and first["lag_1"] == 23
    assert first["lag_24"] == 0 and first["mean_24"] == 11.5
    # An unseen target cannot change the features in its own row.
    changed = y.copy()
    changed.iloc[24] = -999
    before = rows.iloc[0].drop("target")
    after = lagged_training_rows(changed).iloc[0].drop("target")
    pd.testing.assert_series_equal(before, after)
    print(rows.head(1).to_string())
    print("Feature alignment checks passed (synthetic illustration).")
