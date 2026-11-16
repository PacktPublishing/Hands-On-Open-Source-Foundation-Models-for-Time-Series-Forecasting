"""Add Chapter 1 to the repository README table, preserving every existing row.

After copying this chapter folder into the cloned repository, run:
    python scripts/add_readme_entry.py
Use --dry-run to show the proposed row without changing anything.
"""
from __future__ import annotations

import argparse
from pathlib import Path

CHAPTER = "Chapter01_Forecasting_Landscape"
ROOT = Path(__file__).resolve().parents[1]
ROW = (
    "| 1. The New Forecasting Landscape — From ARIMA to Foundation Models | "
    f"[`{CHAPTER}`]({CHAPTER}/) | "
    "Time-series foundations, reproducible teaching figures, and an ETTh1 Forecasting Zoo "
    "comparing AutoARIMA, N-HiTS and zero-shot Chronos-T5-Small with MAE, sMAPE and CRPS |"
)


def add_entry(readme: Path, *, dry_run: bool = False) -> bool:
    """Insert the chapter row once; refuse an unrecognised README table."""
    original = readme.read_bytes()
    text = original.decode("utf-8")
    if f"]({CHAPTER}/)" in text:
        print("Chapter 1 is already linked; README unchanged.")
        return False
    lines = text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines)
               if [part.strip() for part in line.strip().strip("|").split("|")]
               == ["Chapter", "Folder", "What you will run"]]
    if len(matches) != 1:
        raise ValueError("Expected exactly one repository chapter table; README unchanged")
    index = matches[0] + 1
    if index >= len(lines):
        raise ValueError("Chapter table has no separator; README unchanged")
    cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
    if len(cells) != 3 or any(not cell or set(cell) - set("-:") for cell in cells):
        raise ValueError("Chapter table separator is not recognised; README unchanged")
    newline = "\r\n" if lines[index].endswith("\r\n") else "\n"
    row = ROW + newline
    if dry_run:
        print(ROW)
        return True
    # Only add one line; all original bytes and rows otherwise remain unchanged.
    if not lines[index].endswith(("\n", "\r")):
        lines[index] += newline
    lines.insert(index + 1, row)
    readme.write_bytes("".join(lines).encode("utf-8"))
    print("Added Chapter 1 to", readme)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=ROOT.parent / "README.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.readme.is_file():
        parser.error("Repository README not found; first copy the chapter into the repository, "
                     "or pass --readme /path/to/README.md")
    return add_entry(args.readme, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
