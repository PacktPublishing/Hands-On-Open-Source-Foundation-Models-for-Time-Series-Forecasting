"""Chapter 3, Listing 3.4: Timestral with past-only and known-future covariates.

Run from the chapter folder in the env-core environment:
    .venv-core/bin/python chapter_listings/timestral_covariates.py
The first run downloads the Timestral-5M weights from Hugging Face.
"""
import pandas as pd
from timestral import TimestralForecaster

# D2: daily temperature and its twelve covariates, indexed by date
df = pd.read_csv("input_timeseries/D2.csv", index_col="date", parse_dates=True)
known = [c for c in df if c.startswith("fourier") or c == "daylength_h"]
past = [c for c in df if c.startswith("wav") or c in ("nao", "ao")]
T, H = 2048, 30                       # context, horizon; last H days held out

def rows(cols, end):                  # (1, variates, steps) from T+H rows back
    return df[cols].to_numpy("float32").T[None, :, -(T + H):end]

model = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device="cpu")
f = model.forecast(rows(["temp"], -H), prediction_length=H,
                   past_covariates=rows(past, -H),     # stop at the forecast origin
                   known_covariates=rows(known, None)) # run on to the horizon end

print(f.quantiles.shape)              # (series, variates, horizon, quantile levels)
print(f.median[0, 0, :5].round(decimals=1))   # median for the first five days
