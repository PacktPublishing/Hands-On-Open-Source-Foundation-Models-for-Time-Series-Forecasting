"""Chapter 3, Listing 3.5: Moirai-MoE forecasting five metal prices jointly.

Run from the chapter folder in the env-moirai environment:
    .venv-moirai/bin/python chapter_listings/moirai_moe_metals.py
The first run downloads the Moirai-MoE (small) weights from Hugging Face.
"""
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

# D3: monthly prices of five metals; the last H months are held out
df = pd.read_csv("input_timeseries/D3.csv", index_col="date", parse_dates=True)
H = 6
context = df.iloc[:-H]

model = MoiraiMoEForecast(
    module=MoiraiMoEModule.from_pretrained("Salesforce/moirai-moe-1.0-R-small"),
    prediction_length=H, context_length=len(context), patch_size=16,
    num_samples=100, target_dim=5,                 # five metals read jointly
    feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
data = ListDataset([{"start": pd.Period("1982-01", "M"),
                     "target": context.to_numpy("float32").T}],   # (metals, months)
                   freq="M", one_dim_target=False)

torch.manual_seed(0)
forecast = next(iter(model.create_predictor(batch_size=1).predict(data)))
print(forecast.samples.shape)                      # (samples, horizon, metals)
print(np.median(forecast.samples, axis=0)[:3].round(0))   # first three months
