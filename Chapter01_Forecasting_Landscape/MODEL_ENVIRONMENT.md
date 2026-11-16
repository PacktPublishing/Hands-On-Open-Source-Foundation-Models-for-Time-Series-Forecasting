# Optional model rerun environment

Use a separate Python **3.12** virtual environment for model execution. The
inspection environment uses a smaller numerical stack to check retained samples;
the model environment preserves the original model-run direct pins. Do not
combine the two requirements files into one environment.

## CPU installation with uv

From the chapter directory, with `uv` installed:

```bash
uv venv .venv-models -p 3.12
uv pip install -p .venv-models -r requirements-models.txt --torch-backend cpu
```

Activate it with `source .venv-models/bin/activate` on Linux/macOS or
`.venv-models\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip check
python scripts/run_zoo.py --models all
```

`uv venv` need not install pip; if the `pip check` command is unavailable, use
`uv pip check -p .venv-models` instead.

The dependency resolution recorded here targets Linux CPU and Python 3.12.
Other operating systems require their own compatible PyTorch wheel; this package
does not claim that those installations have been tested.

## pip alternative on Linux CPU

```bash
python3.12 -m venv .venv-models
source .venv-models/bin/activate
python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-models.txt
python -m pip check
python scripts/run_zoo.py --models all
```

The installed `2.15.0+cpu` wheel satisfies the `torch==2.15.0` public-version pin.
Weights download from Hugging Face on first Chronos inference; set
`HF_HUB_DISABLE_XET=1` if the optional Xet download path is unavailable. No API key
is required for the public checkpoint. The runner retrieves a missing dataset
from its immutable source revision, checks the checksum, and caches the verified
CSV. An existing checksum mismatch causes an error instead of silent replacement.

## Environment evidence

| File | Meaning |
|---|---|
| `requirements-models.txt` | Original numerical/model direct pins plus notebook and test tooling |
| `lock/model-run-direct.txt` | Preserved original direct requirements; not a transitive lock |
| `results/manifest.json` | Versions and configurations recorded during the original model run |
| `lock/models-linux-cpu-resolved.txt` | New dependency resolution from 2026-11-16; not an installation or training record |
| `lock/inspection-py312.txt` | Installed environment used for this package's lightweight checks |

The model dependency metadata resolves successfully to 160 packages with
PyTorch `2.15.0+cpu`. Full model installation and training were **not rerun during
packaging**. Original saved predictions remain the evidence for the printed
scores. Direct pins alone do not freeze every transitive dependency or guarantee
bit-for-bit reproduction across machines; retained sample arrays allow exact
score verification independently of model reruns.
