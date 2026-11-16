# Chapter 1 — The New Forecasting Landscape

**From ARIMA to Foundation Models**

Companion code for *Hands-On Open-Source Foundation Models for Time Series
Forecasting*, by Tanujit Chakraborty and Shovon Sengupta, Packt.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Hands-On-Open-Source-Foundation-Models-for-Time-Series-Forecasting/blob/main/Chapter01_Forecasting_Landscape/01_Forecasting_Zoo.ipynb)

Start with **[01_Forecasting_Zoo.ipynb](01_Forecasting_Zoo.ipynb)**. It walks through
the recorded ETTh1 experiment, checks its scores, displays forecast intervals, and
explores lag features and the chapter's synthetic illustrations. Its default path
uses bundled outputs and does not download weights or train models. The Colab
badge becomes usable after this folder is pushed to the repository.

## Quick start: inspect the chapter's results

Use Python **3.12**. From this chapter folder:

```bash
python3.12 -m venv .venv-ch01
source .venv-ch01/bin/activate
python -m pip install -r requirements-inspect.txt
python scripts/rebuild_results.py
python -m pytest -q
python -m jupyterlab 01_Forecasting_Zoo.ipynb
```

On Windows, create the environment with `py -3.12 -m venv .venv-ch01` and activate
it in PowerShell with `.venv-ch01\Scripts\Activate.ps1`. In VS Code or Cursor,
open this chapter folder and select the `.venv-ch01` notebook kernel.

After installation, the default notebook and verification commands work offline.
`rebuild_results.py` verifies the saved artifacts without modifying them. The
original execution record is `results/manifest.json`; packaging checks and their
limitations are in [VALIDATION.md](VALIDATION.md).

## Folder guide

| Path | Purpose |
|---|---|
| `01_Forecasting_Zoo.ipynb` | Guided, executed notebook; start here |
| `src/ch01_forecasting/` | Model runners, metrics, feature examples, figure generation, GIFT aggregation |
| `scripts/` | Command-line entry points and repository README helper |
| `code/run_zoo.py` | Compatibility with the command printed in Chapter 1 |
| `chapter_listings/` | Short N-HiTS and Chronos excerpts used in the manuscript |
| `data/ETTh1.csv` | Unmodified, checksum-verified source dataset |
| `results/` | Frozen forecasts, samples, metrics, GIFT snapshot, and provenance |
| `figures/` | Chapter figures in PNG and vector PDF formats |
| `tests/` | Scoring, alignment, artifact integrity, path, and run-isolation checks |
| `requirements*.txt`, `lock/` | Direct dependencies and environment records |
| `licenses/`, `THIRD_PARTY_NOTICES.md` | Dataset, benchmark, and weight provenance |
| `runs/` | New experiments and regenerated artifacts; ignored by git |

The folder is self-contained and does not import from Chapter 3. The complete
manuscript, contracts, and sample books are not part of this code companion.

## Repeat model training and inference

This is optional. Use a **separate** Python 3.12 environment and the model
installation instructions in [MODEL_ENVIRONMENT.md](MODEL_ENVIRONMENT.md).
Weights download from Hugging Face on first use and remain in its cache. The
runner uses CPU explicitly; a GPU is not required. The original run used two
PyTorch threads, but elapsed time and memory use depend on the machine.

```bash
python scripts/run_zoo.py --models all
# The command printed in the book remains supported:
python code/run_zoo.py --models all
```

Run either command once; they perform the same experiment. Each invocation creates
a new `runs/zoo_<timestamp>_<id>/` directory containing its own `results/`,
`figures/`, and execution manifest. It does not overwrite the frozen book results.

Use `--models arima`, `--models nhits`, or `--models chronos` to run one model.
For a named destination, add `--output-dir runs/my-experiment`; it must be new or
empty. A failed model is recorded as failed and excluded from that invocation's
comparison. Scores from older runs are never silently combined with new scores.

N-HiTS uses 200 optimization steps by default. `--max-steps` changes that budget
and therefore changes the experiment. Chronos's optional
`--chronos-sample-batch 8` reduces generation batch size; it can change the random
draws and was not used for the recorded results. Its chunk medians are formed from
all 128 pooled samples before extending the context. Treat either change as a new
experiment, not exact reproduction of the printed numbers.

## What the lab measures

The target is ETTh1 **OT** (oil temperature, degrees Celsius), sampled hourly.
Every model forecasts the same **96-hour** horizon at exclusive origin indices
**11520, 11616, and 11712**: three non-overlapping windows, 288 scored targets.
The original Informer split ends are 8640 / 11520 / 14400, using fixed 30-day
blocks rather than exact calendar months. The lab is univariate; other ETTh1
columns are not used as predictors. Dataset timestamps have no timezone attached.

| Model | Fit and context | Predictive output |
|---|---|---|
| AutoARIMA | Refit on all observations preceding each origin; daily seasonal period 24; bounded stepwise search | Native Normal marginal parameters, sampled for common scoring |
| N-HiTS–Normal | Input length 192; training targets before 8640; validation through 11520 for monitoring; fixed 200-step budget; weights frozen during evaluation | Normal location and scale learned by negative log likelihood |
| Chronos-T5-Small | Original pinned `amazon/chronos-t5-small`; frozen weights; preceding 512 observations | 128 generated samples; native 64-step horizon extended to 96 |

N-HiTS standardizes each input window using its available past observations. No
external scaler is fitted to future targets. The models use different training
regimes and history lengths; this is a bounded teaching comparison, not a tuned
benchmark or a universal ranking.

| Model | MAE | sMAPE (%) | CRPS | 80% coverage (%) | 80% width |
|---|---:|---:|---:|---:|---:|
| AutoARIMA | 1.837 | 18.96 | 1.330 | 95.14 | 8.672 |
| N-HiTS–Normal | 2.021 | 20.75 | 1.364 | 81.25 | 6.587 |
| Chronos-T5-Small | 2.209 | 22.91 | 1.754 | 33.68 | 2.600 |

These are the preserved measured values in `results/metrics.csv`. MAE, CRPS, and
interval width are in degrees Celsius. MAE and sMAPE use the Normal median
(location) or Chronos sample median. sMAPE uses a factor of 200 and assigns zero
when both actual and point forecast are zero. CRPS is the empirical-ensemble
estimator including diagonal pairs, averaged over marginal targets; it is not
the fair-ensemble correction. All methods use 128 samples per origin, and intervals
are empirical sample quantiles without post-hoc calibration. Coverage is a
descriptive result on these windows, not a coverage guarantee.

The Normal samples are independent marginal draws, not coherent temporal paths.
For Chronos, the first chunk's sample median is fed into the next chunk: joint
uncertainty is not fully propagated beyond 64 steps. ETTh1 pretraining overlap is
unknown. Zero-shot inference here means unchanged checkpoint weights; it does
not establish that ETTh1 was absent from pretraining.

## Verify or regenerate figures and tables

```bash
# Read-only checks against retained samples and targets:
python scripts/metrics.py
python scripts/feature_example.py
python scripts/rebuild_results.py
python scripts/gift_reference.py

# Optional regeneration into new, ignored directories:
python scripts/rebuild_results.py --output-dir runs/rebuilt-zoo
python scripts/make_teaching_figures.py --output-dir runs/teaching-figures
python scripts/gift_reference.py --output-dir runs/rebuilt-gift
```

The forecast figure uses actual retained model outputs. Decomposition,
stationarity, learning-regime, and uncertainty illustrations are synthetic or
conceptual and are labelled accordingly. Their parameters and seed are recorded
in `figures/manifest.json`. Regeneration can differ in font rendering or PDF
metadata across environments.

The GIFT table reaggregates **published** scores for four selected methods over
97 aligned configurations at the pinned source revision. It uses geometric means
of model/seasonal-naive score ratios. Its weighted quantile loss is distinct from
this lab's sample CRPS. Upstream overlap flags are retained; no overlap filtering
is applied. This operation does not rerun GIFT models or claim to be a live
leaderboard. Use `--refresh` only together with a new `--output-dir` to fetch that
same pinned source revision again.

## Check and add this folder to the book repository

See [UPLOAD_TO_GITHUB.md](UPLOAD_TO_GITHUB.md) for the author upload steps and the
small additive change needed in the repository's root README. From this chapter
folder in a checkout, the existing repository runner can execute the notebook:

```bash
python ../tools/run_notebooks.py 01_Forecasting_Zoo.ipynb --out runs/notebook-check
```

Code follows the repository's MIT licence. Data, benchmark source files, and
checkpoint weights retain their separate licences; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). `references.bib` retains the
supplied Chapter 1 bibliography for readers following the manuscript's sources.
