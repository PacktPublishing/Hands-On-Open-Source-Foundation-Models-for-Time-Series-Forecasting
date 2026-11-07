"""Run one zero-shot model over every dataset, horizon and window, and store its forecasts."""
# --- cell: runner ---
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ch03_zeroshot.data import (YEAR_STEPS, build_inputs, context_cap, input_plan, load_frame, load_manifest, to_long,
                                validation_rows)
from ch03_zeroshot.models import get_model
from ch03_zeroshot.models.base import CannotForecast

# a covariate dataset whose covariates the model cannot read gives exactly the same input
# as its plain twin, so the plain forecasts are reused (and labelled as such)
TWIN = {"D2": "D1", "D4": "D3"}


SEED = 0


def quiet_logs(level=logging.WARNING):
    """Keep the INFO messages of the model libraries out of the notebook outputs. IBM's granite-tsfm (TTM,
    FlowState, PatchTST-FM) switches Python's root logger to INFO when it is imported, after which every library
    logs its routine steps (down to each glyph of a saved PDF); warnings and errors still show."""
    os.environ.setdefault("TSFM_PYTHON_LOGGING_LEVEL", logging.getLevelName(level))
    root = logging.getLogger()
    if not root.handlers:   # a handler of our own also makes later logging.basicConfig calls do nothing
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(level)
    for handler in root.handlers:
        handler.setLevel(level)


def seed_everything(seed):
    """Fix NumPy's and PyTorch's random generators (sampled forecasts can still differ in the last digits across
    machines and thread counts)."""
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def count_parameters(model, depth=3):
    """Number of weights in the loaded network(s) of an adapter (each tensor counted once)."""
    try:
        import torch
    except ImportError:
        return None
    # the network the adapter loaded, then anything the adapter built later (TTM and Lag-Llama
    # build theirs per horizon, so run_model counts again after forecasting)
    seen, total, stack = set(), 0, [(model, 1), (model._model, 0)]
    visited = set()
    while stack:
        obj, d = stack.pop()
        if id(obj) in visited or obj is None:
            continue
        visited.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            for prm in obj.parameters():
                if id(prm) not in seen:
                    seen.add(id(prm))
                    total += prm.numel()
            continue
        if d < depth and hasattr(obj, "__dict__"):
            stack += [(v, d + 1) for v in vars(obj).values() if hasattr(v, "__dict__")]
    return total or None


def capabilities(model):
    return {"multivariate": model.supports_multivariate, "past_covariates": model.supports_past_covariates,
            "future_covariates": model.supports_future_covariates}


def run_model(key, datasets, root, smoke=False, model=None, verbose=True):
    """Forecast every split of ``datasets`` with model ``key``; returns the run log.

    Writes results/forecasts/<dataset>/<key>.csv and results/runs/<key>.json.
    smoke=True runs only the fixed-origin window of the two shortest horizons (a quick check).
    """
    root = Path(root)
    out_dir = root / "results" / ("forecasts_smoke" if smoke else "forecasts")
    model = model or get_model(key)
    t_load = time.perf_counter()
    model.load()
    log = {"model": key, "name": model.name, "checkpoint": model.checkpoint,
           "load_seconds": round(time.perf_counter() - t_load, 1), "parameters": count_parameters(model),
           "datasets": {}}
    for name in datasets:
        seed_everything(SEED)      # models that sample (Toto-1.0, Moirai-1.1/MoE, Sundial, Lag-Llama) start alike
        plan = input_plan(root, name, capabilities(model))
        entry = {"covariates_used": plan["covariates_used"], "joint": plan["joint"],
                 "known_future": plan["known_future"], "past_only": plan["past_only"], "dropped": plan["dropped"]}
        target = out_dir / name / f"{key}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        twin = out_dir / TWIN.get(name, "-") / f"{key}.csv"
        if name in TWIN and not plan["covariates_used"] and twin.exists():
            df = pd.read_csv(twin).assign(dataset=name)
            df.to_csv(target, index=False)
            entry.update(status=f"same input as {TWIN[name]} (no usable covariates): forecasts reused", seconds=0.0)
            log["datasets"][name] = entry
            continue
        frame = load_frame(root, name)
        manifest = load_manifest(root, name)
        model.set_season_length(YEAR_STEPS[name])   # only FlowState uses it
        if smoke:
            manifest = manifest[manifest["primary"] & manifest["horizon"].isin(sorted(manifest["horizon"].unique())[:2])]
        cap = context_cap(plan, model.max_context)
        parts, seconds, peak, skipped = [], 0.0, 0.0, []
        for h, rows in manifest.groupby("horizon"):
            if model.max_horizon and h > model.max_horizon:
                skipped.append(f"H={h}: longer than the model's maximum horizon {model.max_horizon}")
                continue
            inputs, keys, tests = build_inputs(frame, rows, plan, cap)
            t0 = time.perf_counter()
            try:
                outs = model.predict(inputs, int(h))
            except CannotForecast as err:
                skipped.append(f"H={h}: {err}")
                continue
            seconds += time.perf_counter() - t0
            peak = max(peak, max((o.meta.get("peak_mem_mb") or 0) for o in outs))
            parts.append(to_long(key, name, outs, keys, tests))
            if verbose:
                print(f"  {name} H={h:>3}: {len(inputs):>3} inputs, {time.perf_counter() - t0:6.1f} s", flush=True)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        bad = int(df["y_pred"].isna().sum()) if len(df) else 0
        df.to_csv(target, index=False)
        notes = sorted(getattr(model, "notes", set()))
        if notes:
            entry["notes"] = notes
            model.notes = set()
        entry.update(status="ok" if not skipped else "partial", seconds=round(seconds, 1),
                     peak_mem_mb=round(peak), context_cap=cap, n_forecast_rows=len(df), nan_forecasts=bad,
                     skipped=skipped)
        log["datasets"][name] = entry
        if verbose:
            print(f"{model.name} on {name}: {seconds:.1f} s, covariates: "
                  f"{plan['known_future'] + plan['past_only'] or 'none'}", flush=True)
    log["parameters"] = log["parameters"] or count_parameters(model)
    log["total_seconds"] = round(sum(d.get("seconds", 0) for d in log["datasets"].values()), 1)
    runs = root / "results" / ("runs_smoke" if smoke else "runs")
    runs.mkdir(parents=True, exist_ok=True)
    previous = runs / f"{key}.json"
    if previous.exists():   # keep the entries of datasets run earlier (e.g. main first, bonus later)
        old = json.loads(previous.read_text())
        old["datasets"].update(log["datasets"])
        log = {**old, **{k: v for k, v in log.items() if k != "datasets"}, "datasets": old["datasets"]}
        log["parameters"] = log["parameters"] or old.get("parameters")
        log["total_seconds"] = round(sum(d.get("seconds", 0) for d in log["datasets"].values()), 1)
    previous.write_text(json.dumps(log, indent=1, default=str))
    return log


def run_summary(root, keys, smoke=False):
    """One row per (model, dataset): seconds, covariates used, status."""
    runs = Path(root) / "results" / ("runs_smoke" if smoke else "runs")
    rows = []
    for key in keys:
        p = runs / f"{key}.json"
        if not p.exists():
            continue
        log = json.loads(p.read_text())
        for name, d in log["datasets"].items():
            rows.append({"model": log["name"], "dataset": name, "seconds": d.get("seconds"),
                         "covariates": ", ".join(d.get("known_future", []) + d.get("past_only", [])) or "none",
                         "joint": d.get("joint"), "status": d.get("status")})
    return pd.DataFrame(rows)


def flowstate_scale_check(root, cases=(("D1", 30), ("D3", 12)), model=None):
    """FlowState's time scale: MAE over five validation windows of one horizon that end at the earliest test
    origin (inside the training data; the test windows play no part), once with the hourly default (24 steps per
    cycle, scale factor 1) and once with one year per cycle (the model card's rule, which run_model uses)."""
    root = Path(root)
    model = model or get_model("flowstate")
    model.load()
    rows = []
    for name, h in cases:
        frame, manifest = load_frame(root, name), load_manifest(root, name)
        plan = input_plan(root, name, capabilities(model))
        val = validation_rows(frame, manifest, h)
        inputs, _, tests = build_inputs(frame, val, plan, context_cap(plan, model.max_context))
        truth = np.concatenate([t.to_numpy(float).ravel() for t in tests])
        for label, steps in (("hourly default", 24), ("one year", YEAR_STEPS[name])):
            model.set_season_length(steps)
            pred = np.concatenate([np.asarray(o.mean, float).ravel() for o in model.predict(inputs, h)])
            rows.append({"dataset": name, "horizon": h, "cycle": f"{label} ({steps} steps)",
                         "scale factor": round(model.scale_factor, 4), "forecasts": len(inputs),
                         "validation windows": f"{val['test_start'].min().date()} to {val['test_end'].max().date()}",
                         "MAE": round(float(np.mean(np.abs(pred - truth))), 3)})
    return pd.DataFrame(rows).set_index(["dataset", "horizon", "cycle"])


def covariate_ablation(root, model, dataset, horizon, groups, seed=0):
    """MAE over five validation windows of one horizon (ending at the earliest test origin, so inside the
    training data) when the model reads only some covariates. The test windows play no part.

    groups maps a label to the covariate names to keep (an empty list keeps none); model is a
    registry key or a loaded adapter.
    """
    import dataclasses

    root = Path(root)
    model = get_model(model) if isinstance(model, str) else model
    model.load()
    plan = input_plan(root, dataset, capabilities(model))
    frame, manifest = load_frame(root, dataset), load_manifest(root, dataset)
    val = validation_rows(frame, manifest, horizon)
    inputs, _, tests = build_inputs(frame, val, plan, context_cap(plan, model.max_context))
    truth = np.concatenate([t.to_numpy(float).ravel() for t in tests])

    def pick(values, names, keep):
        idx = [i for i, n in enumerate(names or []) if n in keep]
        return (np.asarray(values)[idx], [names[i] for i in idx]) if idx else (None, None)

    rows = []
    for label, keep in groups.items():
        sub = []
        for s in inputs:
            past, past_names = pick(s.past_covariates, s.past_cov_names, keep)
            fut, fut_names = pick(s.future_covariates, s.future_cov_names, keep)
            sub.append(dataclasses.replace(s, past_covariates=past, past_cov_names=past_names,
                                           future_covariates=fut, future_cov_names=fut_names))
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        pred = np.concatenate([np.asarray(o.mean, float).ravel() for o in model.predict(sub, horizon)])
        rows.append({"covariates": label, "number": len(keep), "MAE": round(float(np.mean(np.abs(pred - truth))), 3),
                     "mean forecast": round(float(pred.mean()), 2), "mean actual": round(float(truth.mean()), 2)})
    return pd.DataFrame(rows).set_index("covariates")
