"""Check frozen empirical evidence and preserve its original provenance."""
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from ch01_forecasting import rebuild_results, run_zoo


def hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for name in ('data', 'results', 'figures')
            for path in (root / name).rglob('*') if path.is_file()}


def test_reference_verification_is_read_only():
    before = hashes(rebuild_results.ROOT)
    report = rebuild_results.main([])
    assert report['status'] == 'passed'
    assert len(report['model_origin_checks']) == 9
    assert 'results/manifest.json' in report['source_sha256']
    assert hashes(rebuild_results.ROOT) == before


def test_regeneration_has_separate_provenance(tmp_path):
    before = hashes(rebuild_results.ROOT)
    output = tmp_path / 'regenerated'
    rebuild_results.main(['--output-dir', str(output)])
    assert (output / 'figures' / 'zoo_forecasts.png').is_file()
    assert (output / 'results' / 'zoo_table.tex').is_file()
    report = json.loads((output / 'results' / 'verification.json').read_text())
    assert report['status'] == 'passed'
    assert 'models not rerun' in report['operation']
    assert not (output / 'results' / 'manifest.json').exists()
    assert hashes(rebuild_results.ROOT) == before


def test_modified_aggregate_is_detected(tmp_path, monkeypatch):
    root = tmp_path / 'copied-chapter'
    for name in ('data', 'results'):
        shutil.copytree(rebuild_results.ROOT / name, root / name)
    path = root / 'results' / 'metrics.csv'
    table = pd.read_csv(path)
    table.loc[0, 'MAE'] += 1
    table.to_csv(path, index=False)
    monkeypatch.setattr(rebuild_results, 'ROOT', root)
    monkeypatch.setattr(run_zoo, 'ROOT', root)
    before = hashes(root)
    with pytest.raises(AssertionError):
        rebuild_results.main([])
    assert hashes(root) == before
