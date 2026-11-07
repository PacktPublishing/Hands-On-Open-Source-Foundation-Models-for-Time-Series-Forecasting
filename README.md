# Hands-On Open-Source Foundation Models for Time Series Forecasting

This is the code repository for *Hands-On Open-Source Foundation Models for Time Series Forecasting: A
Practitioner's Guide to Zero-Shot, Fine-Tuned, and Agentic Forecasting with Python* by Shovon Sengupta and Tanujit
Chakraborty, published by Packt.

The book takes you from familiar statistical, machine-learning and deep-learning baselines to open-source time
series foundation models: zero-shot forecasting, fine-tuning, probabilistic forecasts and their calibration,
explainability, causal variable selection, an agentic forecasting pipeline and deployment. Every chapter comes with
notebooks that run on Google Colab and on your own computer, pinned environments and public datasets.

## How this repository is organised

Each chapter has its own top-level folder, and each folder stands on its own: its notebooks, its shared Python code
(`src/`), its pinned requirement files, its tests and a README that explains how to run it. No chapter imports code
from another, so you can work on one chapter without installing anything for the others.

| Chapter | Folder | What you will run |
|---|---|---|
| 3. Zero-Shot Forecasting in Practice | [`Chapter03_Zero_Shot_Forecasting`](Chapter03_Zero_Shot_Forecasting/) | 23 open foundation models and 16 statistical, machine-learning and deep-learning baselines on six public datasets, evaluated with 14 metrics, MCB tests and Murphy diagrams |

The other chapters are added here as they are completed.

```
Hands-On-Open-Source-Foundation-Models-for-Time-Series-Forecasting/
├── README.md                          this file
├── LICENSE                            MIT licence of the code
├── MAINTAINING.md                     for the authors: adding, checking and publishing a chapter
├── tools/run_notebooks.py             runs notebooks without opening them and reports which ones pass
└── ChapterNN_<Topic>/                 one self-contained folder per chapter
    ├── README.md                      what the chapter's notebooks do, how to run them, results, licences
    ├── NN_*.ipynb, bonus/             the notebooks, numbered in reading order
    ├── src/                           the chapter's shared code (the notebooks import it)
    ├── requirements*.txt, requirements/   pinned environments (lock/ has the full resolved lists)
    ├── input_timeseries/, data/       the frozen datasets and their manifests
    ├── results/                       stored forecasts, tables and figures from the full runs
    └── tests/                         pytest checks
```

## Get the code

```bash
git clone https://github.com/PacktPublishing/Hands-On-Open-Source-Foundation-Models-for-Time-Series-Forecasting.git
cd Hands-On-Open-Source-Foundation-Models-for-Time-Series-Forecasting
```

The *Code > Download ZIP* button on GitHub gives you the same files without git.

## Run the notebooks

**On Google Colab.** Open a notebook on GitHub and click its *Open in Colab* badge, or open it in Colab with
*File > Open notebook > GitHub*. Then choose *Runtime > Run all*. The first code cell clones this repository into the
Colab session, moves into the chapter's folder and installs only what that notebook needs; you do not need to clone
anything yourself. A few notebooks need a particular Python version: the chapter README and the notebook's header
say which and how to pick it in Colab.

**On your own computer (Cursor, VS Code or Jupyter).** You need git, Python 3.12 and
[uv](https://docs.astral.sh/uv/getting-started/installation/) (or `python3.12 -m venv` and pip). The foundation-model
libraries pin different versions of `torch`, `numpy` and `transformers`, so each chapter README has a table of
environments: which requirement file each notebook needs. Create the environment of the notebook you want inside the
chapter folder, for example for Chapter 3:

```bash
cd Chapter03_Zero_Shot_Forecasting
uv venv .venv-dataprep -p 3.12
uv pip install -p .venv-dataprep -r requirements-dataprep.txt
.venv-dataprep/bin/python -m pytest -q          # optional: the chapter's tests
```

Then open the chapter folder itself in your editor (*File > Open Folder* > `Chapter03_Zero_Shot_Forecasting`), not
the repository root: Cursor and VS Code only offer the environments of the folder you open. Pick the
`.venv-dataprep` kernel, not Anaconda's `base` or another Python on your machine, and run the notebook. The `.venv*`
folders are ignored by git.

On Linux without an NVIDIA GPU, add `--torch-backend cpu` to `uv pip install` for environments that contain `torch`:
it installs the CPU build of the same version and skips about 5 GB of CUDA libraries. On macOS this is not needed.

**Check a whole chapter in one go.** `tools/run_notebooks.py` runs notebooks from top to bottom in the environment
that calls it and prints which ones pass, for example:

```bash
cd Chapter03_Zero_Shot_Forecasting
.venv-dataprep/bin/python ../tools/run_notebooks.py 0[0-4]_*.ipynb 11_*.ipynb 12_*.ipynb
```

## Hardware

Unless a chapter README says otherwise, every notebook runs on a CPU, including Colab's free tier. A GPU is used
automatically when one is present. Model checkpoints download from Hugging Face the first time a notebook uses them
and are cached in `~/.cache/huggingface`.

## Licences

The code is released under the MIT licence ([LICENSE](LICENSE)). The datasets and the pretrained model weights keep
their own licences, listed in each chapter's README; some model weights do not allow commercial use.
