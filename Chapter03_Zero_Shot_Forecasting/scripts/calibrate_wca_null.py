"""Null calibration of Notebook 02's keep / weak rule (about 25 minutes on 4 cores).

For every (target, exogenous) pair of D4 and D2, simulate independent AR(1) pairs with the
lag-1 coefficients of the real inputs and the same length, run the same FDR-corrected coherence
test, and record the largest significant share among the bands that reach the forecast horizons.
Notebook 02 reads the result, results/tables/table_wca_null_calibration.csv, and sets the keep and
weak thresholds to its 95th and 80th percentiles (set RECALIBRATE = True there to recompute it
inside the notebook instead).

Usage, from the chapter folder:  python scripts/calibrate_wca_null.py [--workers 4]
"""
import argparse
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")      # one thread per worker process

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ch03_dataprep.config import DATASETS, DRIVERS, HORIZONS, METALS, ROLLING_WINDOWS  # noqa: E402
from ch03_dataprep.splits import insample_end, make_splits  # noqa: E402
from ch03_dataprep.wavelet import ar1_coefficient, null_best_shares  # noqa: E402

BANDS = {"D4": [("2-6 m", 2, 6), ("6-12 m", 6, 12), ("12-24 m", 12, 24), ("24-48 m", 24, 48), ("48-96 m", 48, 96)],
         "D2": [("2-8 d", 2, 8), ("8-32 d", 8, 32), ("32-128 d", 32, 128), ("128-512 d", 128, 512), (">512 d", 512, 1e9)]}
SETTINGS = {"D4": {"pairs": 14, "surrogates": 200, "shortest": 3},     # 15 combinations x 14 = 210 null pairs
            "D2": {"pairs": 20, "surrogates": 50, "shortest": 7}}      # 2 combinations x 20 = 40 null pairs


def inputs():
    """The stationary inputs of Notebook 02, over its in-sample period."""
    out = {}
    d4 = pd.read_csv(ROOT / "input_timeseries/D4.csv", index_col=0, parse_dates=True)
    d4 = d4.loc[:insample_end(make_splits(d4.index, "D4", "monthly", HORIZONS["monthly"], ROLLING_WINDOWS))]
    tgt = pd.DataFrame({m: np.log(d4[m]).diff() for m in METALS}).iloc[1:]
    exo = pd.DataFrame({"global_activity": d4["global_activity"], "usd": np.log(d4["usd"]).diff(),
                        "real_rate_10y": d4["real_rate_10y"].diff()}).iloc[1:]
    out["D4"] = (tgt, exo, list(DRIVERS))
    d2 = pd.read_csv(ROOT / "input_timeseries/D2.csv", index_col=0, parse_dates=True)
    d2 = d2.loc[:insample_end(make_splits(d2.index, "D2", "daily", HORIZONS["daily"], ROLLING_WINDOWS))]
    out["D2"] = (d2[["temp"]], d2[["nao", "ao"]], ["nao", "ao"])   # day length is deterministic: no AR(1) null
    return out


def job(args):
    ds, target, exog, phi_x, phi_y, n, k = args
    s = SETTINGS[ds]
    best = null_best_shares(phi_x, phi_y, n, BANDS[ds], s["shortest"], 1, n_surrogates=s["surrogates"],
                            seed=zlib.crc32(f"{ds}|{target}|{exog}|{k}".encode()))
    return {"dataset": ds, "target": target, "exogenous": exog, "phi_exog": phi_x, "phi_target": phi_y,
            "n": n, "rep": k, "best_share": float(best[0])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    jobs = []
    for ds, (tgt, exo, exogs) in inputs().items():
        for target in tgt:
            for exog in exogs:
                phi_x, phi_y = ar1_coefficient(exo[exog].to_numpy()), ar1_coefficient(tgt[target].to_numpy())
                jobs += [(ds, target, exog, phi_x, phi_y, len(tgt), k) for k in range(SETTINGS[ds]["pairs"])]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(job, jobs, chunksize=1))
    out = pd.DataFrame(rows)
    path = ROOT / "results/tables/table_wca_null_calibration.csv"
    out.to_csv(path, index=False)
    for ds, g in out.groupby("dataset"):
        q = g["best_share"].quantile([0.8, 0.95])
        print(f"{ds}: {len(g)} null pairs; 80th percentile {q[0.8]:.3f}, 95th {q[0.95]:.3f}; "
              f"share >= 10 %: {np.mean(g['best_share'] >= 0.10):.1%}")
    print("saved", path)
