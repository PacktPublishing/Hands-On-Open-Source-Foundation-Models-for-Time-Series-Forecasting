"""Protect frozen references while exercising real figure and score regeneration."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil

import pytest

from ch01_forecasting import gift_reference as gift
from ch01_forecasting import make_teaching_figures as teaching

ROOT = Path(__file__).resolve().parents[1]


def digest_tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for name in ("data", "results", "figures")
            for p in (root / name).rglob("*") if p.is_file()}


def test_gift_default_is_offline_and_read_only(monkeypatch):
    before = digest_tree(ROOT)
    monkeypatch.setattr(gift, "urlopen", lambda *a, **k: pytest.fail("Default must be offline"))
    table = gift.main([])
    assert table["configurations"].tolist() == [97] * 4
    assert table.loc[0, "relative_MASE"] == table.loc[0, "relative_WQL"] == 1.0
    assert digest_tree(ROOT) == before


def test_gift_source_tampering_is_detected(tmp_path):
    dest = tmp_path / "sources"
    shutil.copytree(ROOT / "results/gift_reference_sources", dest)
    target = dest / "Auto_Arima_all_results.csv"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        gift.verify_sources(dest, gift.reference_manifest(ROOT))


def test_gift_reaggregation_requires_matching_saved_table(tmp_path):
    table = gift.aggregate_sources(ROOT / "results/gift_reference_sources")
    altered = table.copy()
    altered.loc[1, "relative_MASE"] += .1
    path = tmp_path / "changed.csv"
    altered.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        gift.verify_table(table, path)


def test_gift_new_copy_preserves_source_provenance(tmp_path):
    before = digest_tree(ROOT)
    dest = tmp_path / "gift-copy"
    gift.main(["--output-dir", str(dest)])
    original = gift.reference_manifest(ROOT)
    copied = json.loads((dest / "gift_reference_manifest.json").read_text())
    assert copied["retrieved_utc_date"] == original["retrieved_utc_date"]
    assert copied["sources"] == original["sources"]
    assert copied["source_revision"] == gift.REVISION
    assert copied["execution_utc"]
    gift.verify_sources(dest / "gift_reference_sources", copied)
    assert digest_tree(ROOT) == before


def test_gift_refresh_requires_separate_output():
    with pytest.raises(SystemExit):
        gift.main(["--refresh"])


def test_gift_refresh_records_new_retrieval_time(tmp_path, monkeypatch):
    before = digest_tree(ROOT)
    cache = ROOT / "results/gift_reference_sources"
    by_url = {s["url"]: (cache / s["file"]).read_bytes()
              for s in gift.reference_manifest(ROOT)["sources"]}
    monkeypatch.setattr(gift, "urlopen", lambda url, **kwargs: io.BytesIO(by_url[url]))
    monkeypatch.setattr(gift, "utc_now", lambda: "2030-02-03T04:05:06+00:00")
    dest = tmp_path / "refreshed"
    gift.main(["--refresh", "--output-dir", str(dest)])
    metadata = json.loads((dest / "gift_reference_manifest.json").read_text())
    assert metadata["retrieved_utc"] == "2030-02-03T04:05:06+00:00"
    assert metadata["retrieved_utc_date"] == "2030-02-03"
    assert metadata["original_retrieved_utc_date"] == "2026-11-14"
    assert digest_tree(ROOT) == before


@pytest.mark.parametrize("protected", ["results", "data", "figures", "results/new"])
@pytest.mark.parametrize("module", [gift, teaching])
def test_outputs_cannot_overlap_bundled_references(protected, module):
    with pytest.raises(ValueError, match="overlaps bundled"):
        module.main(["--output-dir", str(ROOT / protected)])


@pytest.mark.parametrize("module", [gift, teaching])
def test_outputs_cannot_reuse_nonempty_directory(tmp_path, module):
    (tmp_path / "sentinel.txt").write_text("keep")
    with pytest.raises(ValueError, match="new or empty"):
        module.main(["--output-dir", str(tmp_path)])
    assert (tmp_path / "sentinel.txt").read_text() == "keep"


def test_teaching_figures_regenerate_in_a_fresh_directory(tmp_path):
    before = digest_tree(ROOT)
    dest = tmp_path / "teaching"
    assert teaching.main(["--output-dir", str(dest)]) == dest
    manifest = json.loads((dest / "manifest.json").read_text())
    names = {"decomposition", "stationarity", "uncertainty", "learning_regimes"}
    assert set(manifest["figures"]) == names
    for name in names:
        assert (dest / f"{name}.pdf").read_bytes().startswith(b"%PDF")
        assert (dest / f"{name}.png").read_bytes().startswith(b"\x89PNG")
    assert manifest["execution_utc"]
    assert digest_tree(ROOT) == before


def test_readme_insertion_is_additive_and_idempotent(tmp_path):
    spec = importlib.util.spec_from_file_location("add_readme_entry", ROOT / "scripts/add_readme_entry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = ("# Book\r\n\r\n| Chapter | Folder | What you will run |\r\n"
                "|---|---|---|\r\n| 3. Chapter Three | existing/ | Keep this row |\r\n\r\nAfter\r\n")
    path = tmp_path / "README.md"
    path.write_bytes(original.encode())
    assert module.add_entry(path, dry_run=True)
    assert path.read_bytes() == original.encode()
    assert module.add_entry(path)
    new = path.read_bytes().decode()
    assert new.replace(module.ROW + "\r\n", "") == original
    assert not module.add_entry(path)
    assert path.read_bytes().decode() == new
