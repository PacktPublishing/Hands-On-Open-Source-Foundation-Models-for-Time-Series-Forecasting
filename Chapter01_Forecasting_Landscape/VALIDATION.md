# Packaging validation — 2026-11-16

The supplied Chapter 1 resources were reorganized for the book repository's
`ChapterNN_<Topic>` layout. The measured book results were retained; packaging
did not rerun model training or inference.

| Check | Outcome |
|---|---|
| Clean Python 3.12.14 inspection environment, pinned requirements | Installed; `pip check` passed |
| Full test suite in a fresh local Git clone | **37 passed** |
| All default notebook code cells, in order in a fresh process | **10 passed**, with three rich figure outputs |
| Frozen data, results, and figure files before/after notebook execution | **39 unchanged**, checked by SHA-256 |
| Notebook and read-only checks in the fresh clone | Git working tree stayed clean |
| Model/origin sample archives | All **9** combinations verified against ETTh1 targets |
| Per-origin and aggregate scores, forecast timestamps and intervals | Recomputed and matched retained artifacts |
| GIFT sources and table | Eight source files hash-verified; 97-configuration aggregation matched |
| Figure generation | Four teaching illustrations regenerated in temporary destinations; frozen figures unchanged |
| New run behavior | Separate outputs, failed-model isolation, selected-model status, nonempty/protected-path rejection tested |
| Root README addition | Additive, idempotent insertion tested; existing rows and CRLF line endings preserved |
| Notebook path discovery | Repository root, chapter directory, and nested directory tested |
| Printed `code/run_zoo.py` command | Compatibility wrapper and CLI help checked |
| Model implementation bodies | AutoARIMA, N-HiTS, Chronos, and Gaussian sampling functions unchanged from supplied code |
| Model dependency metadata | Linux CPU/Python 3.12 resolution passed for 160 packages |

## Execution method and limits

The notebook was evaluated using a fresh-process **IPython InteractiveShell**,
with all cells sharing one clean namespace and rich display outputs captured
into the `.ipynb` file. The notebook metadata records this execution method.

The repository's ordinary Jupyter/nbclient runner was attempted, but the execution
environment rejected both TCP and local IPC socket binds before any notebook
code ran. Therefore normal Jupyter kernel startup and hosted Google Colab
execution are **not claimed as tested**. The public notebook uses standard
Python/Jupyter code and contains no environment-specific workaround. Readers
can run the repository's normal notebook runner on their own machine.

The fresh clone was a **local test repository** containing this chapter and the
inspected root README, ignore rules, licence, maintenance guide, and notebook
runner. It was not a published GitHub branch. The Colab badge is prepared for
`main` and becomes live after the author pushes/merges Chapter 1.

No checkpoint weights were downloaded during these packaging checks. The full
model environment was resolved from package metadata, not installed and run.
The optional smaller Chronos generation batch remains outside the original
executed experiment. These limits are separate from the earlier model execution
recorded in `results/manifest.json`.

## Preserved evidence

The 39 retained files under `data/`, `results/`, and `figures/` are byte-identical
to their supplied Chapter 1 counterparts. The original execution and source
manifests were not rewritten to describe this packaging session. New model
runs and figure/table regeneration use separate directories and provenance.

For a local recheck, install `requirements-inspect.txt` and follow the commands in
`README.md` and `UPLOAD_TO_GITHUB.md`. Full model reruns are optional and use
`MODEL_ENVIRONMENT.md`.
