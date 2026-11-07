"""Run notebooks from top to bottom without opening them, and report which ones pass.

Use it to check a chapter after a fresh ``git clone``: each notebook runs in the Python environment that runs
this script (the environment needs ``nbclient`` and ``ipykernel``; every chapter's requirement files include
them), from the notebook's own folder, exactly as when you choose *Run all* in Jupyter, Cursor or VS Code.

    cd Chapter03_Zero_Shot_Forecasting
    .venv-core/bin/python ../tools/run_notebooks.py 05a_*.ipynb 05b_*.ipynb bonus/B05a_*.ipynb

By default the executed notebooks are thrown away, so the files you cloned stay as they are (the notebooks may
still write their own outputs, such as quick-check forecasts, into the chapter's results folder). Options:

    --out DIR       save each executed notebook in DIR (to read the outputs or an error in context)
    --inplace       save each executed notebook over the original (for authors refreshing the stored outputs)
    --timeout S     seconds a single cell may take (default 7200)
    --report FILE   append one JSON line per notebook (name, status, seconds, error) to FILE

The exit code is the number of notebooks that failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def kernel_for_this_python() -> str:
    """Register a temporary Jupyter kernel that runs this very Python, and return its name."""
    home = Path(tempfile.mkdtemp(prefix="run-notebooks-"))
    spec = home / "kernels" / "this-python"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": f"Python ({sys.executable})",
        "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = os.pathsep.join(filter(None, [str(home), os.environ.get("JUPYTER_PATH")]))
    return "this-python"


def first_error(nb) -> str:
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                return f"{out.get('ename')}: {out.get('evalue')}"[:500]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("notebooks", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError
    except ImportError:
        sys.exit("This environment has no nbclient: install it with `uv pip install nbclient ipykernel` "
                 "(or pip), or run the script with the Python of the notebook's environment.")
    kernel = kernel_for_this_python()
    failed = 0
    print(f"Python: {sys.executable}")
    for path in args.notebooks:
        path = path.resolve()
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=args.timeout, kernel_name=kernel,
                                resources={"metadata": {"path": str(path.parent)}})
        start = time.time()
        status, error = "passed", ""
        try:
            client.execute()
        except CellExecutionError as exc:
            status, error = "failed", first_error(nb) or str(exc).strip().splitlines()[-1][:500]
        except Exception as exc:   # kernel died, timeout, ...
            status, error = "failed", f"{type(exc).__name__}: {exc}"[:500]
        seconds = round(time.time() - start, 1)
        failed += status == "failed"
        print(f"{status:6}  {seconds:8.1f} s  {path.name}" + (f"\n        {error}" if error else ""), flush=True)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, args.out / path.name)
        if args.inplace:
            nbformat.write(nb, path)
        if args.report:
            with open(args.report, "a") as fh:
                fh.write(json.dumps({"notebook": str(path), "status": status, "seconds": seconds,
                                     "error": error, "python": sys.executable}) + "\n")
    print(f"{len(args.notebooks) - failed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
