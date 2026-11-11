# Chapter 3 · Zero-Shot Forecasting in Practice

Code for *Hands-On Open-Source Foundation Models for Time Series Forecasting* (Packt), Chapter 3.

The chapter follows one experiment from start to finish: build and describe the datasets, check which covariates
should help and how forecastable each series is, fix the train/test splits (Notebooks 00 to 04), forecast every
split zero-shot with 23 open foundation models (Notebooks 05a to 09b) and with 16 statistical, machine-learning and
deep-learning baselines (Notebook 10), then evaluate the point forecasts with 14 metrics in paper-style tables
(Notebook 11) and test how robust the ranking is with MCB tests and Murphy diagrams (Notebook 12). The bonus
datasets B1 and B2 have their own copies of Notebooks 05a to 12 in [`bonus/`](bonus/). The chapter stops at point
forecasts; probabilistic evaluation comes later in the book.

## Notebooks

**Data and pre-forecast analysis**

| Notebook | What it does | Run time | Peak memory |
|---|---|---|---|
| [00 · Data acquisition and dictionary](00_data_acquisition_and_dictionary.ipynb) | Downloads the public sources, builds the six datasets, checks them, writes the snapshot and the data dictionary | under 1 minute | 0.2 GB |
| [01 · Global characteristics](01_global_characteristics_and_summary_statistics.ipynb) | Summary statistics, shape, stationarity, non-linearity, seasonality, long memory, breaks, co-movement | under 1 minute | 0.8 GB |
| [02 · Exogenous relevance](02_exogenous_relevance.ipynb) | Wavelet coherence of each candidate covariate with its target; linear, non-linear and scale-by-scale Granger causality; SURD; rolling correlation; an explicit keep/weak/drop expectation per covariate, tested later by running the top models with and without covariates | about 7 minutes | 1.2 GB |
| [03 · Forecastability](03_forecastability.ipynb) | Why forecastability is checked first; about fifteen measures, each explained and read against a shuffled-noise reference; a monthly-averaging simulation; one profile and plain-language story per series; an expected-difficulty ranking | about 3 minutes | 1.4 GB |
| [04 · Train and test splits](04_train_test_splits.ipynb) | Fixed- and rolling-origin splits, boundary rows, leakage and checkpoint-date checks, the split manifest | under 1 minute | 0.2 GB |

**Forecasting, evaluation and robustness**

| Notebook | Models | Environment | Quick check (default) | Full run, D1 to D4 | Full run, B1 and B2 |
|---|---|---|---|---|---|
| [05a · Chronos, TimesFM and TiRex](05a_zero_shot_chronos_timesfm_tirex.ipynb) | Chronos-2, Chronos-Bolt (small), TimesFM-2.5, TimesFM-3, TiRex, TiRex-2 | `env-core` | 10 min | 6 min | 6 min |
| [05b · Timestral, IBM's models, t0 and TabPFN-TS](05b_zero_shot_timestral_ibm_t0_tabpfn.ipynb) | Timestral-209M, Timestral-5M, TTM-r2, FlowState-r1.1, PatchTST-FM-r2, t0-gamma, TabPFN-TS (v2 weights) | `env-core` | 2 min | 72 min | 35 min |
| [06 · the Moirai family](06_zero_shot_moirai.ipynb) | Moirai-1.1 (small), Moirai-MoE (small), Moirai-2.0 (small) | `env-moirai` | 7.5 s | 42 min | 44 min |
| [07a · Toto-2.0](07a_zero_shot_toto2.ipynb) | Toto-2.0 (22m) | `env-toto2` | 0.2 s | 1.2 s | 3.8 s |
| [07b · Toto-1.0](07b_zero_shot_toto1.ipynb) | Toto-1.0 | `env-toto1` | 18 s | 9 min | 7 min |
| [08 · Sundial, Timer and Time-MoE](08_zero_shot_sundial_timer_timemoe.ipynb) | Sundial (base), Timer (base), Time-MoE (50M) | `env-thuml` | 14 s | 4 min | 19 min |
| [09a · Kairos](09a_zero_shot_kairos.ipynb) | Kairos (50M) | `env-kairos` | 2.5 s | 4.5 s | 9.4 s |
| [09b · Lag-Llama](09b_zero_shot_lag_llama.ipynb) | Lag-Llama | `env-lagllama` | 8.8 s | 15 min | 83 min |
| [10 · Baselines](10_baselines.ipynb) | Naive (RW), Seasonal naive, RWD, AR, ARIMA, ARIMAX, ARDL, VAR, VARX; LightGBM, XGBoost, CatBoost, NHiTS, TFT, DLinear, TiDE | `env-baselines` | 4 min | 84 min | 39 min |
| [11 · Evaluation](11_evaluation.ipynb) | every stored forecast: 14 metrics, paper-style tables, horizons, covariates, scorecard | `requirements-dataprep.txt` | about 1 min | (reads stored results) | |
| [12 · Robustness](12_robustness_mcb_murphy.ipynb) | MCB tests and Murphy diagrams for the top models | `requirements-dataprep.txt` | about 1 min | (reads stored results) | |

Bonus notebooks (B1 WTI crude oil, B2 US Treasury yields): [B05a](bonus/B05a_zero_shot_chronos_timesfm_tirex.ipynb), [B05b](bonus/B05b_zero_shot_timestral_ibm_t0_tabpfn.ipynb), [B06](bonus/B06_zero_shot_moirai.ipynb), [B07a](bonus/B07a_zero_shot_toto2.ipynb), [B07b](bonus/B07b_zero_shot_toto1.ipynb), [B08](bonus/B08_zero_shot_sundial_timer_timemoe.ipynb), [B09a](bonus/B09a_zero_shot_kairos.ipynb), [B09b](bonus/B09b_zero_shot_lag_llama.ipynb),
[B10](bonus/B10_baselines.ipynb), [B11](bonus/B11_evaluation.ipynb), [B12](bonus/B12_robustness_mcb_murphy.ipynb).

Times are forecasting time on a 4-core CPU without a GPU; model loading and downloads come on top. The quick
checks ran one notebook at a time on all four cores. The full runs used one CPU thread per notebook while up to
five notebooks shared the machine, so they are upper bounds for a notebook run alone. Colab's free tier has 2 cores,
so expect the quick checks to take up to twice as long there. **Every forecasting notebook runs a quick check by default** (`FULL_RUN = False`: the fixed-origin window
at the two shortest horizons of each dataset) and analyses the stored full results in `results/forecasts/`. Set
`FULL_RUN = True` at the top of a notebook to recompute its full results. No notebook needs a GPU.

**Run order.** Notebooks 05a to 10 are independent of each other: run any of them, in any order, each in its own
environment. Notebooks 11 and 12 read the stored forecasts of all of them, so they can run straight away; run them
again after you recompute anything.

**Recomputing the full results.** The stored results come from a full run of every notebook from a fresh clone of
this repository (11 November 2026, the CPU machine described above). A full rerun reproduces the stored forecasts of
every model that does not draw samples, up to rounding (the largest gap in our check was 0.12% on one TiRex forecast): Chronos-2, Chronos-Bolt,
TimesFM, TiRex, Timestral, TTM, FlowState, PatchTST-FM, t0, TabPFN-TS, Moirai-2.0, Toto-2.0, Kairos, Timer, Time-MoE
and the statistical and tree-based baselines. Models that draw sample paths (Lag-Llama, Toto-1.0, Moirai-1.1,
Moirai-MoE, Sundial) and the neural baselines trained here (NHiTS, TFT, TiDE) start from a fixed seed but give
different forecasts on a different machine (on the same machine our reruns of Sundial and Lag-Llama matched). Their scores then
move, and with them every model's mean rank and close calls such as which models the MCB test cannot tell apart from
the best. An earlier run on another machine, for example, gave Lag-Llama a mean MASE of 0.909 on D1 (0.935 here), TFT
0.921 (1.073 here), and 11 models in the pooled MCB group (8 here). The book quotes the stored results, which are what
the notebooks show by default. Times depend on the machine too. To run
several notebooks at once, set `CH03_THREADS=1` and `OMP_NUM_THREADS=1` first (multi-threaded libraries slow down
badly on a shared CPU), and allow about 7 GB of memory for each of Notebooks 05b and B05b (TabPFN-TS forecasts in
several worker processes).

## Which models can use covariates zero-shot

"Zero-shot" means the weights are used as released, with no training on our data. Some models can still read
covariates in their context; the classes below were checked against each paper or model card and against the
library version pinned in `requirements/` (a test ties every row to the adapter's code).

| Model | Organisation | Several targets jointly | Past-only covariates | Known-future covariates | How | Source |
|---|---|---|---|---|---|---|
| Chronos-2 | Amazon | yes | yes | yes | group attention across target and covariate tokens | Ansari et al. (2025), arXiv:2510.15821 |
| Chronos-Bolt (small) | Amazon | no | no | no | univariate only | Ansari et al. (2024), arXiv:2403.07815; model card |
| TimesFM-2.5 | Google | no | no | yes | XReg: in-context linear regression on the covariates, fitted per series | Das et al. (2024), arXiv:2310.10688; timesfm 2.x README |
| TimesFM-3 | Google | yes | yes | yes | variate attention over target and covariate channels | model card google/timesfm-3.0-pytorch (no paper yet) |
| TiRex | NX-AI | no | no | no | univariate only | Auer et al. (2025), arXiv:2505.23719 |
| TiRex-2 | NX-AI | yes | yes | yes | asymmetric group attention; reverse xLSTM pass for known-future inputs | Podest, Pichler et al. (2026), arXiv:2607.01204 |
| Timestral-209M | Mistral AI | yes | yes | yes | variate attention every 4th block; covariates cannot read targets | Mistral AI (2026), arXiv:2610.37585 |
| Timestral-5M | Mistral AI | yes | yes | yes | as Timestral-209M | Mistral AI (2026), arXiv:2610.37585 |
| TTM-r2 | IBM | no | no | no | exogenous mixer exists but only after fine-tuning; zero-shot is univariate | Ekambaram et al. (2024), arXiv:2401.03955 |
| FlowState-r1.1 | IBM | no | no | no | univariate only | Graf et al. (2025), arXiv:2508.05287 |
| PatchTST-FM-r2 | IBM | no | no | no | univariate only | model card ibm-granite/granite-timeseries-patchtst-fm-r2 |
| t0-gamma | The Forecasting Company | yes | yes | yes | alternating time and variate attention; known-future rows read bidirectionally | Meyer et al. (2026), arXiv:2609.24559 |
| TabPFN-TS (v2 weights) | Prior Labs | no | no | yes | known-future covariates become extra feature columns of the regression table | Hoo et al. (2025), arXiv:2501.02945 |
| Moirai-1.1 (small) | Salesforce | yes | yes | yes | all variates flattened into one sequence; binary variate attention bias | Woo et al. (2024), arXiv:2402.02592 |
| Moirai-MoE (small) | Salesforce | yes | yes | yes | same packing and attention bias as Moirai-1.1 | Liu, Liu et al. (2024), arXiv:2410.10469 |
| Moirai-2.0 (small) | Salesforce | no | no | no | univariate: the paper dropped multivariate and covariate support | Liu, Aksu et al. (2025), arXiv:2511.11698, Sections 3.1 and 6 |
| Toto-2.0 (22m) | Datadog | yes | no | no | joint targets via variate attention; exogenous inputs not yet supported | Khwaja et al. (2026), arXiv:2605.20119; toto-2 README |
| Toto-1.0 | Datadog | yes | no | yes | known-future covariates passed as extra variates (exogenous API); their future values reach only steps 65 onward | Cohen et al. (2025), arXiv:2505.14766; toto-ts 0.2.0 |
| Sundial (base) | Tsinghua THUML | no | no | no | univariate only | Liu, Qin et al. (2025), arXiv:2502.00816 |
| Timer (base) | Tsinghua THUML | no | no | no | univariate checkpoint (Timer-XL adds covariates only when trained for them) | Liu et al. (2024), arXiv:2402.02368; model card |
| Time-MoE (50M) | Princeton, Squirrel Ai, Griffith et al. | no | no | no | univariate only | Shi et al. (2025), arXiv:2409.16040 |
| Kairos (50M) | ShanghaiTech and Ant Group | no | no | no | univariate only | Feng et al. (2025), arXiv:2509.25826 |
| Lag-Llama | Morgan Stanley, ServiceNow, Mila et al. | no | no | no | univariate only (calendar features are built from timestamps, not passed in) | Rasul et al. (2023), arXiv:2310.08278 |

Considered and left out: **MOMENT** (no zero-shot forecasting head: the forecasting head must be trained first); **YingLong** (the released inference path needs a CUDA GPU (FlashAttention)); **Timer-S1** (GPU-only inference in the released code); **TempoPFN** (GPU-only inference in the released code); **TabPFN-TS default weights (TabPFN-2.5)** (gated: needs a Prior Labs account and token; the ungated v2 weights are used).

How covariates enter the runs: D2 and D4 give every model all the covariates it can read (Notebook 02's verdicts are
expectations, not filters); a model that reads none gets exactly the D1 or D3 input, so its D1/D3 forecasts are
reused for D2/D4 and labelled as such. Only calendar covariates (D2's Fourier terms and day length, functions of the
date) are known over the test window; every other covariate is past-only and stops at the forecast origin, so no
forecast ever reads a value from its test window (`covariate_lists` raises an error otherwise).

## Run design

| Setting | Value |
|---|---|
| Horizons | 7, 14, 30, 60, 90 days (D1, D2, B1, B2); 3, 6, 9, 12, 24 months (D3, D4) |
| Test windows | 5 rolling origins per daily horizon, 3 per monthly horizon (1 for 24 months); `data/splits/` |
| Context | the last 2,048 observations (daily) or the full history (monthly), shortened only where a model's limit is smaller |
| Point forecast | the median of each model's forecast distribution (the mean for models that only give one) |
| Several targets | forecast jointly by models that can (D3, D4, B2), one by one otherwise |
| CPU settings | Toto-1.0 32 samples (library default 256); Sundial 20 samples (model card); TabPFN-TS 2 estimators (default 8); Moirai-MoE shortens D2's context so its 13 variates fit 640 tokens |
| Per-model inputs | FlowState's time scale is set to one year per cycle (`scale_factor` = 24 / steps per year, the model card's rule); Lag-Llama gets the real dates for its calendar features; TTM left-pads (and masks) monthly contexts shorter than its 512 steps; Time-MoE forecasts at most five series per call (memory) |
| Randomness | models that sample their forecasts (Toto-1.0, Moirai-1.1, Moirai-MoE, Sundial, Lag-Llama) and the learned baselines start from a fixed seed; on another machine their forecasts change, which usually moves their scores in the second decimal and sometimes more (see *Recomputing the full results*) |
| Baselines | statistical models refitted at every origin on the same context, their orders chosen there by AIC or AICc; learned models (all in Darts) tuned by a short Optuna search that starts from the default setting, scored on validation windows at the end of the training data, then trained once on the data before the first test window (GPU when present) |

## Environments

Foundation-model libraries pin different versions of `torch`, `numpy` and `transformers`, so the notebooks are
grouped by the environment they need. Every file in `requirements/` is pinned; `requirements/lock/` has the full
resolved lists.

| Environment | Notebooks | Python | Extra install after the requirements file |
|---|---|---|---|
| [`env-core`](requirements/env-core.txt) | 05a, 05b | 3.12 (tested) or 3.13 | none |
| [`env-moirai`](requirements/env-moirai.txt) | 06 | 3.12 only | none |
| [`env-toto2`](requirements/env-toto2.txt) | 07a | 3.12 (tested) or 3.13 | none |
| [`env-toto1`](requirements/env-toto1.txt) | 07b | 3.12 only | `uv pip install --no-deps toto-ts==0.2.0` |
| [`env-thuml`](requirements/env-thuml.txt) | 08 | 3.12 only | none |
| [`env-kairos`](requirements/env-kairos.txt) | 09a | 3.12 (tested) or 3.13 | `uv pip install "git+https://github.com/foundation-model-research/Kairos@16eb4b9"` |
| [`env-lagllama`](requirements/env-lagllama.txt) | 09b | 3.12 only | `uv pip install --no-deps "git+https://github.com/time-series-foundation-models/lag-llama@df7531a"` |
| [`env-baselines`](requirements/env-baselines.txt) | 10 | 3.12 (tested) or 3.13 | none |
| [`requirements-dataprep.txt`](requirements-dataprep.txt) | 00 to 04, 11, 12 | 3.12 or 3.13 | none |

**Google Colab.** Open a notebook with its *Open in Colab* badge and choose *Runtime > Run all*. The first code cell
fetches this folder and installs the notebook's environment with [uv](https://docs.astral.sh/uv/), which Colab
comes with and which is several times faster than pip. Four environments (`moirai`, `toto1`, `thuml`,
`lagllama`) need Python 3.12: in Colab choose *Runtime > Change runtime type* and pick a runtime version with Python
3.12 first; the setup cell says so if you forget. If the cell says it is restarting the session, wait, then choose
*Runtime > Run all* again. To run your own copy from Google Drive, set `DRIVE_COPY` in that cell to the folder's path.

**Your own computer (Cursor, VS Code or Jupyter).** Clone the book's repository, move into this chapter's folder
and, with [uv](https://docs.astral.sh/uv/) (or `python3.12 -m venv` and `pip`), create the environment of the
notebook you want, for example:

```bash
git clone https://github.com/PacktPublishing/Hands-On-Open-Source-Foundation-Models-for-Time-Series-Forecasting.git
cd Hands-On-Open-Source-Foundation-Models-for-Time-Series-Forecasting/Chapter03_Zero_Shot_Forecasting
uv venv .venv-core -p 3.12 && uv pip install -p .venv-core -r requirements/env-core.txt
uv venv .venv-dataprep -p 3.12 && uv pip install -p .venv-dataprep -r requirements-dataprep.txt
```

then open this folder itself in your editor (*File > Open Folder* > `Chapter03_Zero_Shot_Forecasting`, not the
repository root, since Cursor and VS Code only offer the environments of the folder you open), select the matching
kernel (`.venv-dataprep` for 00 to 04, 11 and 12, `.venv-core` for 05a and 05b, and so on; not Anaconda's `base`) and
run all cells. The extra installs in the table above go after the requirements file
(`uv pip install -p .venv-toto1 --no-deps toto-ts==0.2.0`, and so on). All environments resolve on Linux and on
Apple-silicon Macs. On Linux without an NVIDIA GPU, add `--torch-backend cpu` to each `uv pip install` that
installs `torch`: you get the CPU build of the same version and skip about 5 GB of CUDA libraries. Model checkpoints
download from Hugging Face on first use (about 9 GB for all 23 models).

To run several notebooks in one go without opening them, use the repository's `tools/run_notebooks.py` with the
environment's Python; it prints which notebooks pass:

```bash
.venv-core/bin/python ../tools/run_notebooks.py 05a_*.ipynb 05b_*.ipynb bonus/B05a_*.ipynb bonus/B05b_*.ipynb
```

**From the command line.** `python scripts/run_zero_shot.py --notebook 05a` (add `--bonus` for B1 and B2,
`--smoke` for the quick check) and `python scripts/run_baselines.py` recompute the stored forecasts in
`results/forecasts/` and the run logs in `results/runs/`.

## Headline results

- **Temperature (D1, D2):** 22 of 23 foundation models beat the seasonal naive rule with no training, and most of them
  cannot be told apart statistically (MCB, Notebook 12). Chronos-Bolt has the lowest mean MASE (0.672, against 0.858
  for the seasonal naive rule); the best tuned baselines, DLinear (0.684) and an ARDL with D2's covariates (0.675),
  come close but do not beat it.
- **Metal prices (D3, D4):** a hard case. Only Chronos-Bolt and PatchTST-FM among the foundation models, and CatBoost,
  the random walk with drift and the VAR among the baselines, beat the random walk on average; CatBoost (2.55) and
  Chronos-Bolt (2.56) are level at the top, and the random walk stays in the MCB winning group. PatchTST-FM has the
  best mean rank over D1 to D4.
- **Covariates:** reading them zero-shot changes most foundation models' error by less than 3% on average;
  Moirai-1.1 and Moirai-MoE get much worse with D2's twelve covariates. MASE and RMSE agree on the direction of the
  change in 172 of 190 cases.
- **Bonus (oil, yields):** near-random-walk series. On oil a few foundation models beat the random walk by under 4%
  (PatchTST-FM best); on yields a tuned CatBoost leads, and Timestral-5M is the only foundation model that beats the
  random walk.
- **No single metric gives the full picture:** on D3 MAE and RMSE pick PatchTST-FM while MASE picks CatBoost, and
  no Murphy curve dominates another anywhere.


Where each model ranks in every (dataset, series, horizon) cell of D1 to D4, rescaled from 0 (best) to 100 (worst)
and averaged, from `results/tables/table_scorecard.csv`:

| Rank | Model | Family | Mean rank, % (MASE) | Mean rank, % (RMSSE) | Best in (cells) | Datasets |
|---|---|---|---|---|---|---|
| 1 | PatchTST-FM-r2 | foundation model | 27.8 | 28.1 | 6 of 60 | D1, D2, D3, D4 |
| 2 | VARX | statistical | 29.3 | 29.7 | 0 of 25 | D4 |
| 3 | Chronos-Bolt (small) | foundation model | 30.7 | 28.8 | 13 of 60 | D1, D2, D3, D4 |
| 4 | VAR | statistical | 34.1 | 33.0 | 1 of 25 | D3 |
| 5 | FlowState-r1.1 | foundation model | 36.4 | 36.2 | 1 of 60 | D1, D2, D3, D4 |
| 6 | RWD | statistical | 37.2 | 36.8 | 0 of 60 | D1, D2, D3, D4 |
| 7 | Moirai-2.0 (small) | foundation model | 37.7 | 38.1 | 1 of 60 | D1, D2, D3, D4 |
| 8 | Naive (RW) | statistical | 38.9 | 37.9 | 1 of 60 | D1, D2, D3, D4 |
| 9 | CatBoost | machine learning | 39.5 | 39.0 | 3 of 60 | D1, D2, D3, D4 |
| 10 | TiRex | foundation model | 40.2 | 42.0 | 1 of 60 | D1, D2, D3, D4 |

## Datasets

| ID | Content | Grain | Rows | Period |
|---|---|---|---|---|
| D1 | Central England daily mean temperature (Met Office HadCET v2.1) | daily | 10,000 | 16 May 1999 to 30 Sep 2026 |
| D2 | D1 + Fourier terms and day length (known in advance), causal wavelet bands, NAO and AO (past only) | daily | 10,000 | as D1 |
| D3 | Copper, aluminium, nickel, zinc and lead (World Bank Pink Sheet, LME monthly averages) | monthly | 535 | Jan 1982 to Jul 2026 |
| D4 | D3 + Kilian global activity index, BIS dollar index, Cleveland Fed 10-year real rate (past only) | monthly | 535 | as D3 |
| B1 | WTI crude oil spot price (EIA via FRED), bonus | trading day | 9,513 | 2 Jan 1986 to 30 Sep 2026 |
| B2 | US Treasury yields, 3 months to 10 years (Federal Reserve H.15 via FRED), bonus | trading day | 10,000 | 8 Oct 1986 to 30 Sep 2026 |

The frozen datasets (vintage 30 September 2026, downloaded 1 November 2026) are CSV files in `input_timeseries/`:
wide form (`D1.csv` ... `B2.csv`, one column per series, indexed by date) and long form (`long/D1_long.csv`, columns
`unique_id`, `ds`, `y`, the layout most forecasting libraries expect). `input_timeseries/snapshot_manifest.json`
records every download with its URL, time and checksum.

**Data dictionary.** `input_timeseries/data_dictionary_ch03.xlsx` has six sheets:

| Sheet | Contents |
|---|---|
| Read me | What the workbook is, the vintage, how the datasets were built |
| Datasets | One row per dataset: grain, calendar, rows, time frame, targets, covariates, units, horizons, rolling windows, files, sources with links, licences, citation, preparation steps |
| Variables | One row per column: description, unit, role (target, known-future or past-only covariate), source link, count, minimum, maximum, mean |
| Sources | One row per downloaded file: publisher, product, series id, landing page, download URL, licence, frequency, unit, download time, size, SHA-256 |
| Splits | One row per train/test split: origin, train and test dates, rows (written by Notebook 04) |
| Outlier flags | Observations flagged for inspection (kept in the data, not removed) |

Horizons are 7, 14, 30, 60 and 90 steps for daily data and 3, 6, 9, 12 and 24 months for monthly data; the splits
are in `data/splits/`.

**Refresh the data.** Notebook 00 downloads the latest data into `data/raw/` and, by default, writes the new
snapshot to `data/rebuild/input_timeseries/` and compares it with the frozen one, so the book's numbers stay
reproducible. Set `OVERWRITE_SNAPSHOT = True` in its first code cell only if you want to replace the book's snapshot.

## Code listings from the chapter

The listings written for the chapter text are in `chapter_listings/`, with the same code as the book. Run them from
this folder in the environment given below; each prints the output shown in the chapter. The other listings in the
chapter are excerpts of `src/`, and their captions name the file.

| Listing | File | What it does | Environment | Run time on a CPU |
|---|---|---|---|---|
| 3.1 | [`first_zero_shot_forecast.py`](chapter_listings/first_zero_shot_forecast.py) | Chronos-Bolt (small) forecasts the next 30 days of D1 from its last 2,048 days, with the 10%, 50% and 90% quantiles (downloads 191 MB of weights the first time) | `env-core` | about 10 s |
| 3.2 | [`causal_attention.py`](chapter_listings/causal_attention.py) | one causal attention head with `nn.MultiheadAttention`, checked against the attention equation | `env-core` | about 1 s |
| 3.3 | [`pinball_loss.py`](chapter_listings/pinball_loss.py) | the pinball (quantile) loss on a two-step example | `env-core` | about 1 s |
| 3.4 | [`timestral_covariates.py`](chapter_listings/timestral_covariates.py) | Timestral-5M forecasts the last 30 days of D2 with its past-only and known-future covariates | `env-core` | about 20 s |
| 3.5 | [`moirai_moe_metals.py`](chapter_listings/moirai_moe_metals.py) | Moirai-MoE (small) forecasts the last six months of the five metal prices of D3 jointly, as 100 sample paths | `env-moirai` | about 1.5 min |

```bash
.venv-core/bin/python chapter_listings/first_zero_shot_forecast.py
.venv-core/bin/python chapter_listings/causal_attention.py
.venv-core/bin/python chapter_listings/pinball_loss.py
.venv-core/bin/python chapter_listings/timestral_covariates.py
.venv-moirai/bin/python chapter_listings/moirai_moe_metals.py
```

The chapter's figure of covariate gains (`results/figures/fig_ch03_covariate_gains.pdf` and `.png`) is drawn from the
stored scores by `.venv-dataprep/bin/python scripts/make_covariate_figure.py`.

## Tests

```bash
.venv-dataprep/bin/python -m pytest -q
```

Run them in the `requirements-dataprep.txt` environment, which has every package they need except `jax`: the one
test that needs it (TimesFM-2.5's covariate settings) is skipped there and passes with
`.venv-core/bin/python -m pytest -q tests/test_adapter_settings.py`. The tests check the wavelet-coherence port,
the covariate features, the splits and the forecastability measures on known cases; the forecast metrics, the MCB test and the Murphy elementary scores on hand-computed examples; that every
number in an architecture figure equals the released model configuration (snapshots in `data/model_configs/`) and
that the covariate table matches each model adapter; and that every notebook's copy of the shared code matches
`src/` and was saved after a clean run.

## Folder layout

```
00_ ... 04_*.ipynb          data and pre-forecast analysis
05a_ ... 09b_*.ipynb        zero-shot foundation models, one notebook per environment
10_baselines.ipynb          statistical, machine-learning and deep-learning baselines
11_evaluation.ipynb         metrics, paper-style tables, horizons, covariates, scorecard
12_robustness_mcb_murphy.ipynb   MCB tests and Murphy diagrams
bonus/                      the same notebooks (05a to 12) for B1 and B2
chapter_listings/           the code listings written for the chapter text (Listings 3.1 to 3.5)
src/ch03_dataprep/          shared code of Notebooks 00 to 04
src/ch03_zeroshot/          shared code of Notebooks 05a to 12: model adapters (models/), runner, data plans,
                            baselines, metrics, tables, robustness, architecture figures (architectures, diagrams)
input_timeseries/           the frozen datasets, snapshot manifest and data dictionary
data/splits/                split manifest and one JSON file per dataset
data/model_checkpoint_dates.csv   dates of each foundation model's weights on Hugging Face
data/model_configs/         released configuration of each model, used to check the architecture figures
results/forecasts/          stored forecasts, one CSV per dataset and model (long format)
results/runs/               run logs: seconds, peak memory, covariates read, notes
results/tables/             metric tables; paper/ holds the paper-style tables as CSV, Excel and LaTeX
results/figures/            figures, including architectures/ (PNG, plus a vector PDF of each for print), robustness/ and robustness_bonus/
requirements/               one pinned file per environment (lock/ has the full resolved lists)
requirements-dataprep.txt   environment of Notebooks 00 to 04, 11 and 12
requirements-colab.txt      the packages Colab lacks for Notebooks 00 to 04
scripts/                    command-line runners (run_zero_shot.py, run_baselines.py), calibrate_wca_null.py,
                            make_covariate_figure.py (a figure of the chapter) and set_repo_path.py
                            (see MAINTAINING.md at the repository root)
tests/                      pytest checks
```

## Data sources and licences

- Met Office Hadley Centre, HadCET v2.1: Open Government Licence v3.0.
- NOAA Climate Prediction Center, daily NAO and AO: US Government work, public domain.
- World Bank Commodity Price Data (the Pink Sheet): CC BY 4.0.
- FRED (Federal Reserve Bank of St. Louis): DCOILWTICO (EIA) and DGS3MO to DGS10 (Federal Reserve Board) are
  public domain; IGREA (Kilian, via the Dallas Fed), NNUSBIS (BIS) and REAINTRATREARAT10Y (Cleveland Fed) are
  redistributed under their publishers' terms, cited in the data dictionary.

**Model weights** have their own licences; check them before any commercial use:

| Model | Checkpoint | Licence of the weights |
|---|---|---|
| Chronos-2 | [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2) | Apache-2.0 |
| Chronos-Bolt (small) | [`amazon/chronos-bolt-small`](https://huggingface.co/amazon/chronos-bolt-small) | Apache-2.0 |
| TimesFM-2.5 | [`google/timesfm-2.5-200m-pytorch`](https://huggingface.co/google/timesfm-2.5-200m-pytorch) | Apache-2.0 |
| TimesFM-3 | [`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch) | timesfm-non-commercial-license-v1.0 (weights); Apache-2.0 (code) |
| TiRex | [`NX-AI/TiRex`](https://huggingface.co/NX-AI/TiRex) | NXAI Community License |
| TiRex-2 | [`NX-AI/TiRex-2`](https://huggingface.co/NX-AI/TiRex-2) | Apache-2.0 |
| Timestral-209M | [`mistralai/Timestral-209M`](https://huggingface.co/mistralai/Timestral-209M) | Modified MIT (weights; no rights above USD 20M monthly revenue); Apache-2.0 (code) |
| Timestral-5M | [`mistralai/Timestral-5M`](https://huggingface.co/mistralai/Timestral-5M) | Modified MIT (weights; no rights above USD 20M monthly revenue); Apache-2.0 (code) |
| TTM-r2 | [`ibm-granite/granite-timeseries-ttm-r2`](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) | Apache-2.0 |
| FlowState-r1.1 | [`ibm-granite/granite-timeseries-flowstate-r1`](https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1) | Apache-2.0 |
| PatchTST-FM-r2 | [`ibm-granite/granite-timeseries-patchtst-fm-r2`](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2) | Apache-2.0 / OpenMDW-1.0 |
| t0-gamma | [`theforecastingcompany/t0-gamma`](https://huggingface.co/theforecastingcompany/t0-gamma) | Apache-2.0 |
| TabPFN-TS (v2 weights) | [`Prior-Labs/TabPFN-v2-reg`](https://huggingface.co/Prior-Labs/TabPFN-v2-reg) | tabpfn-time-series: Apache-2.0; weights: Prior Labs licence (check before commercial use) |
| Moirai-1.1 (small) | [`Salesforce/moirai-1.1-R-small`](https://huggingface.co/Salesforce/moirai-1.1-R-small) | CC-BY-NC-4.0 |
| Moirai-MoE (small) | [`Salesforce/moirai-moe-1.0-R-small`](https://huggingface.co/Salesforce/moirai-moe-1.0-R-small) | CC-BY-NC-4.0 |
| Moirai-2.0 (small) | [`Salesforce/moirai-2.0-R-small`](https://huggingface.co/Salesforce/moirai-2.0-R-small) | CC-BY-NC-4.0 |
| Toto-2.0 (22m) | [`Datadog/Toto-2.0-22m`](https://huggingface.co/Datadog/Toto-2.0-22m) | Apache-2.0 |
| Toto-1.0 | [`Datadog/Toto-Open-Base-1.0`](https://huggingface.co/Datadog/Toto-Open-Base-1.0) | Apache-2.0 |
| Sundial (base) | [`thuml/sundial-base-128m`](https://huggingface.co/thuml/sundial-base-128m) | Apache-2.0 |
| Timer (base) | [`thuml/timer-base-84m`](https://huggingface.co/thuml/timer-base-84m) | Apache-2.0 |
| Time-MoE (50M) | [`Maple728/TimeMoE-50M`](https://huggingface.co/Maple728/TimeMoE-50M) | Apache-2.0 |
| Kairos (50M) | [`mldi-lab/Kairos_50m`](https://huggingface.co/mldi-lab/Kairos_50m) | Apache-2.0 |
| Lag-Llama | [`time-series-foundation-models/Lag-Llama`](https://huggingface.co/time-series-foundation-models/Lag-Llama) | Apache-2.0 |
