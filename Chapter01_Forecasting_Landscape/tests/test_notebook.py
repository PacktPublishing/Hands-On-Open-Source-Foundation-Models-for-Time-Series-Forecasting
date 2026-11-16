"""The notebook opens from a checkout, a chapter, or a nested directory."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


CHAPTER = "Chapter01_Forecasting_Landscape"
NOTEBOOK = Path(__file__).resolve().parents[1] / "01_Forecasting_Zoo.ipynb"


@pytest.mark.parametrize("working_directory", ["repository", "chapter", "nested"])
def test_notebook_setup_finds_relocated_chapter(tmp_path, monkeypatch, working_directory):
    checkout = tmp_path / "an-arbitrarily-named-checkout"
    chapter = checkout / CHAPTER
    (chapter / "results").mkdir(parents=True)
    (chapter / "results/manifest.json").write_text("{}")
    (chapter / "src/ch01_forecasting").mkdir(parents=True)
    (chapter / "src/ch01_forecasting/metrics.py").write_text("")
    directory = {"repository": checkout, "chapter": chapter,
                 "nested": chapter / "src/ch01_forecasting"}[working_directory]
    monkeypatch.chdir(directory)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *args, **kwargs: None if name == "google.colab"
                        else find_spec(name, *args, **kwargs))
    setup = json.loads(NOTEBOOK.read_text())["cells"][1]
    namespace = {}
    exec(compile("".join(setup["source"]), str(NOTEBOOK), "exec"), namespace)
    assert namespace["ROOT"] == chapter.resolve()
    assert namespace["IN_COLAB"] is False
    assert str(chapter / "src") in sys.path
