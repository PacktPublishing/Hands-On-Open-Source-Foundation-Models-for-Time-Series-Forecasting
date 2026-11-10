"""Chapter 3, Listing 3.1: a first zero-shot forecast with Chronos-Bolt.

Run from the chapter folder in the env-core environment:
    .venv-core/bin/python chapter_listings/first_zero_shot_forecast.py
The first run downloads the Chronos-Bolt (small) weights from Hugging Face (191 MB).
"""
import pandas as pd
import torch
from chronos import BaseChronosPipeline

# Central England daily mean temperature (dataset D1), indexed by date
df = pd.read_csv("input_timeseries/D1.csv", index_col="date", parse_dates=True)
context = torch.tensor(df["temp"].to_numpy()[-2048:])     # last 2,048 days

# Download the released weights once; no training follows
pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small",
                                               device_map="cpu")
quantiles, median = pipeline.predict_quantiles(   # Bolt returns its median
    [context], prediction_length=30, quantile_levels=[0.1, 0.5, 0.9])

print(quantiles.shape)          # (series, horizon, quantile levels)
print(quantiles[0, :5].round(decimals=1))   # first five days: q10, median, q90
