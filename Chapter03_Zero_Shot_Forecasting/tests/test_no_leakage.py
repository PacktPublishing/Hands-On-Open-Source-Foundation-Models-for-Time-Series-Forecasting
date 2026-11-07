"""No test data in any forecast (Shovon, 4 Nov 2026): neither target values nor covariate values from a test window
may reach a model, its fitting, its tuning or any check that chooses a setting."""
import json
from pathlib import Path

import numpy as np
import pytest

from ch03_zeroshot import data
from ch03_zeroshot.baselines import BASELINES, STATISTICAL, VAL_WINDOWS, _Learned, caps, runs_on

ROOT = Path(__file__).resolve().parents[1]
ALL = data.MAIN + data.BONUS
EVERYTHING = {"multivariate": True, "past_covariates": True, "future_covariates": True}


@pytest.mark.parametrize("name", ALL)
def test_only_calendar_covariates_are_known_future(name):
    known, past = data.covariate_lists(ROOT, name)
    assert set(known) <= set(data.CALENDAR_COVARIATES)
    assert not set(known) & set(past)


def test_a_non_calendar_known_future_covariate_is_refused(tmp_path):
    snap = json.loads((ROOT / "input_timeseries" / "snapshot_manifest.json").read_text())
    snap["covariates"]["D4"] = {"known_future": ["global_activity"], "past_only": []}
    (tmp_path / "input_timeseries").mkdir()
    (tmp_path / "input_timeseries" / "snapshot_manifest.json").write_text(json.dumps(snap))
    with pytest.raises(ValueError, match="not calendar covariates"):
        data.covariate_lists(tmp_path, "D4")


@pytest.mark.parametrize("name", ALL)
def test_model_inputs_stop_at_the_origin(name):
    """Every input a foundation model or statistical baseline gets: the target and past-only covariates end at the
    origin; only calendar covariates continue over the test window."""
    frame, manifest = data.load_frame(ROOT, name), data.load_manifest(ROOT, name)
    plan = data.input_plan(ROOT, name, EVERYTHING)
    inputs, keys, tests = data.build_inputs(frame, manifest, plan)
    for s, (g, h, _, origin), test in zip(inputs, keys, tests):
        t = np.shape(s.target)[-1]
        assert frame.index[t - 1] == origin and test.index.min() > origin
        np.testing.assert_array_equal(np.atleast_2d(s.target)[:, -1], frame.loc[origin, g].to_numpy(float))
        if s.past_covariates is not None:
            assert np.shape(s.past_covariates)[-1] == t
        if s.future_covariates is not None:
            assert set(s.future_cov_names) <= set(data.CALENDAR_COVARIATES)
            assert np.shape(s.future_covariates)[-1] == t + h


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("h", [12, 30, 90])
def test_validation_windows_end_by_the_first_test_origin(name, h):
    frame, manifest = data.load_frame(ROOT, name), data.load_manifest(ROOT, name)
    val = data.validation_rows(frame, manifest, h)
    first = manifest["origin_date"].min()
    assert (val["test_end"] <= first).all() and (val["n_test"] == h).all()
    assert val["test_end"].max() == first          # back to back, the last one ends at the earliest origin


@pytest.mark.parametrize("name", ALL)
def test_learned_baselines_train_and_tune_before_the_first_test_window(name):
    frame, manifest = data.load_frame(ROOT, name), data.load_manifest(ROOT, name)
    for key in BASELINES:
        if key in STATISTICAL or not runs_on(key, name):
            continue
        learner = _Learned(key, name, frame, manifest, data.input_plan(ROOT, name, caps(key)))
        assert frame.index[learner.train_end - 1] == manifest["origin_date"].min()
        val_start = learner.train_end - VAL_WINDOWS * learner.st["out"]
        assert learner.start < val_start < learner.train_end


def test_stored_searches_used_only_training_data():
    stored = sorted((ROOT / "results" / "tuning").glob("*/*.json"))
    if not stored:
        pytest.skip("no stored hyperparameter searches")
    for p in stored:
        t = json.loads(p.read_text())
        first = data.load_manifest(ROOT, t["dataset"])["origin_date"].min()
        v = t["validation"]
        assert v["earliest_test_origin"] == str(first.date()), p
        assert v["first_date"] <= v["last_date"] <= v["earliest_test_origin"], p
        if t.get("default_val_MASE") is not None:          # Optuna's trial 0 is the default setting
            first_trial = t["trials"][0]["params"]
            assert first_trial == {k: t["defaults"][k] for k in first_trial}, p
