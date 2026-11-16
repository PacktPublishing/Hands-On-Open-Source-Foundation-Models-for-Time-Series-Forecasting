"""Output isolation and failure handling without training or model downloads."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ch01_forecasting import artifacts, run_zoo


@pytest.fixture
def stubbed_zoo(tmp_path, monkeypatch):
    root = tmp_path / 'renamed-chapter'
    root.mkdir()
    monkeypatch.setattr(run_zoo, 'ROOT', root)
    frame = pd.DataFrame({'unique_id': 'ETTh1_OT',
                          'ds': pd.date_range('2020-01-01', periods=run_zoo.TEST_END, freq='h'),
                          'y': np.zeros(run_zoo.TEST_END)})
    monkeypatch.setattr(run_zoo, 'load_data', lambda: frame)
    monkeypatch.setattr(run_zoo, 'draw_forecasts', lambda *args, **kwargs: None)
    outputs = [(origin, np.ones((run_zoo.N_SAMPLES, run_zoo.HORIZON)),
                np.ones(run_zoo.HORIZON)) for origin in run_zoo.ORIGINS]
    monkeypatch.setattr(run_zoo, 'arima_forecasts', lambda frame: (outputs, {'test_stub': True}))
    return root


def test_single_model_run_is_fresh_and_not_a_complete_comparison(stubbed_zoo, monkeypatch):
    # Recording an absent unrelated package must not block a selected model.
    def missing(_):
        raise artifacts.importlib.metadata.PackageNotFoundError('not installed')
    monkeypatch.setattr(artifacts.importlib.metadata, 'version', missing)
    first = run_zoo.main(['--models', 'arima'])
    second = run_zoo.main(['--models', 'arima'])
    assert first != second
    assert first.parent == stubbed_zoo / 'runs'
    manifest = json.loads((first / 'results' / 'manifest.json').read_text())
    assert manifest['status'] == 'executed'
    assert manifest['requested_models'] == ['AutoARIMA']
    assert manifest['successful_models'] == ['AutoARIMA']
    assert manifest['selected_models_complete'] is True
    assert manifest['comparison_complete'] is False
    assert manifest['packages']['chronos-forecasting'] is None
    assert set(manifest['models']) == {'AutoARIMA'}


def test_failed_model_files_never_enter_current_run_aggregation(stubbed_zoo, tmp_path, monkeypatch):
    output = tmp_path / 'partial-run'
    def fail_with_stale_output(*args):
        # Simulate a partial failed write; a directory glob would include it.
        stale = pd.DataFrame({'model': ['NHITS-Normal'], 'origin_index': [11520], 'MAE': [999.]})
        stale.to_csv(output / 'results' / 'nhits_normal_scores.csv', index=False)
        raise RuntimeError('deliberate model failure')
    def fail(*args):
        raise RuntimeError('deliberate model failure')
    monkeypatch.setattr(run_zoo, 'nhits_forecasts', fail_with_stale_output)
    monkeypatch.setattr(run_zoo, 'chronos_forecasts', fail)
    with pytest.raises(SystemExit, match='Failures recorded'):
        run_zoo.main(['--models', 'all', '--output-dir', str(output)])
    metrics = pd.read_csv(output / 'results' / 'metrics.csv')
    assert metrics.model.tolist() == ['AutoARIMA']
    assert metrics.MAE.tolist() == [1.0]
    manifest = json.loads((output / 'results' / 'manifest.json').read_text())
    assert manifest['status'] == 'partial'
    assert manifest['successful_models'] == ['AutoARIMA']
    assert manifest['selected_models_complete'] is False
    assert manifest['comparison_complete'] is False
    assert manifest['models']['NHITS-Normal']['status'] == 'failed'


def test_existing_run_cannot_be_silently_reused(stubbed_zoo, tmp_path):
    output = run_zoo.main(['--models', 'arima', '--output-dir', str(tmp_path / 'run')])
    manifest = output / 'results' / 'manifest.json'
    original = manifest.read_bytes()
    with pytest.raises(SystemExit):
        run_zoo.main(['--models', 'arima', '--output-dir', str(output)])
    assert manifest.read_bytes() == original


@pytest.mark.parametrize('relative', ['data', 'results', 'figures', 'data/new', 'results/new', 'figures/new', '.','..'])
def test_protected_output_overlap_is_rejected(tmp_path, relative):
    root = tmp_path / 'chapter'
    root.mkdir()
    with pytest.raises(ValueError, match='overlaps bundled'):
        artifacts.fresh_output_dir(root / relative, root=root)


def test_protected_output_symlink_is_rejected(tmp_path):
    root = tmp_path / 'chapter'
    (root / 'results').mkdir(parents=True)
    link = tmp_path / 'alias'
    link.symlink_to(root / 'results', target_is_directory=True)
    with pytest.raises(ValueError, match='overlaps bundled'):
        artifacts.fresh_output_dir(link / 'new', root=root)


def test_nonpositive_training_steps_rejected_before_running(stubbed_zoo):
    with pytest.raises(SystemExit):
        run_zoo.main(['--max-steps', '0'])
    assert not (stubbed_zoo / 'runs').exists()
