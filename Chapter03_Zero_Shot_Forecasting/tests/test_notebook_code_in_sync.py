"""The notebooks show the shared code they run: each cell tagged `module_cell` must equal its block in src/ch03_dataprep."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob("0[0-4]_*.ipynb"))   # 05a to 12 are checked in test_zeroshot_notebooks_in_sync.py


def module_blocks(module):
    text = (ROOT / "src" / "ch03_dataprep" / f"{module}.py").read_text()
    parts = re.split(r"^# --- cell: ([\w-]+) ---\n", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}


def test_all_five_notebooks_present():
    assert [p.name[:2] for p in NOTEBOOKS] == ["00", "01", "02", "03", "04"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_module_cells_match_source(path):
    nb = json.loads(path.read_text())
    tagged = [c for c in nb["cells"] if c.get("metadata", {}).get("module_cell")]
    assert tagged, f"{path.name} has no module cells"
    for cell in tagged:
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
