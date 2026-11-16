"""Run the Chapter 1 feature_example module from any working directory."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":
    runpy.run_module("ch01_forecasting.feature_example", run_name="__main__")
