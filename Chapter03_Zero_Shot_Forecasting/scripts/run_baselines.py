"""Run the baselines of Notebook 10 from the command line.

    python scripts/run_baselines.py                      # every baseline on D1 D2 D3 D4
    python scripts/run_baselines.py --models arima tft   # some of them
    python scripts/run_baselines.py --bonus              # B1 and B2
    python scripts/run_baselines.py --smoke              # fixed origin, two horizons only
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ch03_zeroshot.baselines import BASELINES, run_baseline  # noqa: E402
from ch03_zeroshot.data import BONUS, MAIN  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="*", default=list(BASELINES))
p.add_argument("--datasets", nargs="*", default=MAIN)
p.add_argument("--bonus", action="store_true")
p.add_argument("--smoke", action="store_true")
a = p.parse_args()
datasets = BONUS if a.bonus else a.datasets
failed = []
for k in a.models:
    t = time.perf_counter()
    print(f"=== {k}", flush=True)
    try:
        run_baseline(k, datasets, ROOT, smoke=a.smoke)
    except Exception:
        traceback.print_exc()
        failed.append(k)
    print(f"=== {k} done in {time.perf_counter() - t:.0f} s", flush=True)
print("FAILED:", failed or "none")
