"""Verify/re-aggregate the chapter's frozen GIFT-Eval scores without running models.

The default is offline and read-only: verify cached source hashes, aggregate the
97 configurations, and compare with the bundled table. Use --output-dir for a
new copy; --refresh additionally retrieves the same immutable upstream revision.
Neither mode replaces the bundled reference files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .artifacts import fresh_output_dir, utc_now

REVISION = "0b57dd6937eb9a69a3ae771f73ab36188bb33e47"
MODELS = ["Seasonal_Naive", "Auto_Arima", "Chronos_small", "chronos_bolt_small"]
METRICS = {
    "relative_MASE": "eval_metrics/MASE[0.5]",
    "relative_WQL": "eval_metrics/mean_weighted_sum_quantile_loss",
}
ROOT = Path(__file__).resolve().parents[2]


def source_url(model: str, name: str) -> str:
    return ("https://raw.githubusercontent.com/SalesforceAIResearch/"
            f"gift-eval/{REVISION}/results/{model}/{name}")


def reference_manifest(root: Path = ROOT) -> dict:
    """Load the frozen provenance and reject missing/unexpected source entries."""
    manifest = json.loads((root / "results/gift_reference_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_revision") != REVISION:
        raise ValueError("The bundled GIFT-Eval manifest has an unexpected source revision")
    expected = {
        (model, f"{model}_{name}", source_url(model, name))
        for model in MODELS for name in ("all_results.csv", "config.json")
    }
    sources = manifest.get("sources", [])
    observed = {(item.get("model"), item.get("file"), item.get("url")) for item in sources}
    if observed != expected or len(sources) != len(expected):
        raise ValueError("The bundled GIFT-Eval source list differs from the frozen illustration")
    for item in sources:
        digest = item.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid source checksum: {item['file']}")
    return manifest


def verify_sources(source_dir: Path, manifest: dict) -> None:
    """Check all eight source files against their recorded SHA-256 digests."""
    for item in manifest["sources"]:
        path = source_dir / item["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen GIFT-Eval source: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"GIFT-Eval source checksum mismatch: {path.name}")


def aggregate_sources(source_dir: Path) -> pd.DataFrame:
    """Geometrically average each model/baseline ratio on the common 97 rows."""
    frames = {}
    for model in MODELS:
        frame = pd.read_csv(source_dir / f"{model}_all_results.csv")
        frame["dataset"] = frame["dataset"].str.strip()
        if frame["dataset"].isna().any() or frame["dataset"].duplicated().any():
            raise ValueError(f"Missing or duplicate configuration for {model}")
        frames[model] = frame.set_index("dataset")
    baseline = frames["Seasonal_Naive"]
    rows = []
    for model, frame in frames.items():
        if set(frame.index) != set(baseline.index) or len(frame) != 97:
            raise ValueError("This frozen illustration requires identical 97 configurations")
        frame = frame.reindex(baseline.index)
        config = json.loads((source_dir / f"{model}_config.json").read_text(encoding="utf-8"))
        row = {"model": model, "configurations": len(frame),
               "source_overlap_flag": config.get("testdata_leakage", "Unknown")}
        for short, source in METRICS.items():
            values = frame[source].to_numpy(dtype=float)
            denominator = baseline[source].to_numpy(dtype=float)
            if not (np.isfinite(values).all() and np.isfinite(denominator).all()
                    and (values > 0).all() and (denominator > 0).all()):
                raise ValueError("Cannot take a geometric mean of nonpositive/nonfinite scores")
            row[short] = float(np.exp(np.log(values / denominator).mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def verify_table(table: pd.DataFrame, reference_csv: Path) -> None:
    """Require the frozen table to match a fresh aggregation, up to CSV rounding."""
    saved = pd.read_csv(reference_csv)
    try:
        pd.testing.assert_frame_equal(table, saved, check_exact=False, rtol=1e-12, atol=1e-14)
    except AssertionError as exc:
        raise ValueError("The bundled GIFT-Eval table does not match its frozen sources") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="Write a new copy to this fresh directory; default only verifies references.")
    parser.add_argument("--refresh", action="store_true",
                        help="Download the pinned revision again; requires --output-dir.")
    args = parser.parse_args(argv)
    if args.refresh and args.output_dir is None:
        parser.error("--refresh requires --output-dir so reference files are never replaced")

    manifest = reference_manifest(ROOT)
    cached = ROOT / "results/gift_reference_sources"
    verify_sources(cached, manifest)
    table = aggregate_sources(cached)
    verify_table(table, ROOT / "results/gift_reference.csv")

    if args.output_dir is not None:
        output_dir = fresh_output_dir(args.output_dir, root=ROOT, prefix="gift")
        dest = output_dir / "gift_reference_sources"
        dest.mkdir()
        for item in manifest["sources"]:
            target = dest / item["file"]
            if args.refresh:
                with urlopen(item["url"], timeout=60) as stream:
                    payload = stream.read()
                if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                    raise ValueError(f"Downloaded frozen source checksum mismatch: {item['file']}")
                target.write_bytes(payload)
            else:
                shutil.copyfile(cached / item["file"], target)
        verify_sources(dest, manifest)
        table = aggregate_sources(dest)
        verify_table(table, ROOT / "results/gift_reference.csv")
        table.to_csv(output_dir / "gift_reference.csv", index=False)
        metadata = dict(manifest)
        metadata["execution_utc"] = utc_now()
        metadata["execution"] = "Frozen published scores verified and re-aggregated; models not rerun"
        metadata["retrieval_mode"] = "downloaded pinned source revision" if args.refresh else "copied verified cached sources"
        if args.refresh:
            metadata["original_retrieved_utc_date"] = manifest.get("retrieved_utc_date")
            metadata["retrieved_utc"] = utc_now()
            metadata["retrieved_utc_date"] = metadata["retrieved_utc"][:10]
        (output_dir / "gift_reference_manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print("Wrote a new GIFT-Eval reference aggregation to", output_dir)
    else:
        print("Verified eight cached GIFT-Eval sources and the saved table (no files changed).")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    main()
