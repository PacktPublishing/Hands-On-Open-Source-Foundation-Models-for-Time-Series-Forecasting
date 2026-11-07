"""Run zero-shot models from the command line (the notebooks call the same function).

    python scripts/run_zero_shot.py --models chronos2 tirex --datasets D1 D2 D3 D4
    python scripts/run_zero_shot.py --notebook 05a            # every model of one notebook
    python scripts/run_zero_shot.py --notebook 05a --smoke    # fixed origin, two horizons only
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ch03_zeroshot.data import BONUS, MAIN  # noqa: E402
from ch03_zeroshot.models import models_in_notebook  # noqa: E402
from ch03_zeroshot.runner import run_model  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="*", default=[])
p.add_argument("--notebook")
p.add_argument("--datasets", nargs="*", default=MAIN)
p.add_argument("--bonus", action="store_true", help="run B1 and B2 instead")
p.add_argument("--smoke", action="store_true")
a = p.parse_args()
keys = a.models or models_in_notebook(a.notebook)
datasets = BONUS if a.bonus else a.datasets
failed = []
for k in keys:
    t = time.perf_counter()
    print(f"=== {k}", flush=True)
    try:
        run_model(k, datasets, ROOT, smoke=a.smoke)
    except Exception:
        traceback.print_exc()
        failed.append(k)
    print(f"=== {k} done in {time.perf_counter() - t:.0f} s", flush=True)
print("FAILED:", failed or "none")
