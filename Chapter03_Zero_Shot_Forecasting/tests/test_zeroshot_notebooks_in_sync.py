"""Notebooks 05a to 12 (and bonus/) show the shared code they run: each cell tagged `module_cell` must equal its
block in src/ch03_zeroshot, and every notebook must be saved after a clean run."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(p for p in ROOT.glob("*.ipynb") if p.name[:2] >= "05") + sorted((ROOT / "bonus").glob("B*.ipynb"))


def module_blocks(module):
    text = (ROOT / "src" / "ch03_zeroshot" / f"{module}.py").read_text()
    parts = re.split(r"^# --- cell: ([\w-]+) ---\n", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}


def test_every_notebook_present():
    names = {p.name.split("_")[0] for p in NOTEBOOKS if p.parent == ROOT}
    assert {"05a", "05b", "06", "07a", "07b", "08", "09a", "09b", "10", "11", "12"} <= names
    assert len([p for p in NOTEBOOKS if p.parent.name == "bonus"]) == 11


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_module_cells_match_source(path):
    nb = json.loads(path.read_text())
    for cell in [c for c in nb["cells"] if c.get("metadata", {}).get("module_cell")]:
        module, name = cell["metadata"]["module_cell"].split(":")
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        assert source == module_blocks(module)[name], f"{path.name}: cell {module}:{name} differs from src/"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_saved_with_outputs_and_no_errors(path):
    nb = json.loads(path.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c.get("execution_count") for c in code), f"{path.name} has cells that were not run"
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"{path.name} was saved with an error output"
