"""Point the Colab badges and the notebooks' clone command at the book's GitHub repository.

The notebooks and the Markdown files name the repository and this chapter's folder in it
(``PacktPublishing/Hands-On-Open-Source-Foundation-Models-for-Time-Series-Forecasting``, folder
``Chapter03_Zero_Shot_Forecasting``, branch ``main``). If the chapter is published under another name,
run this once from the chapter folder before committing::

    python scripts/set_repo_path.py --folder Chapter03                    # another folder name
    python scripts/set_repo_path.py --repo PacktPublishing/New-Name       # another repository
    python scripts/set_repo_path.py --folder Chapter03 --dry-run          # only list the changes

It reads the current repository, branch and folder from the Colab badge of Notebook 00, then
rewrites the badge links, the ``https://github.com/<repository>`` URLs, the ``CHAPTER = "<folder>"``
line of every notebook's setup cell (the folder the notebook moves into after cloning) and every other
mention of the folder name, and prints each file it changed.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BADGE = re.compile(r"colab\.research\.google\.com/github/([\w.-]+/[\w.-]+)/blob/([\w.-]+)/([\w.-]+)/")
SKIP = {".git", "results", "data", "__pycache__", ".ipynb_checkpoints"}


def current() -> tuple[str, str, str]:
    """(repository, branch, folder) of the first Colab badge in README.md or, failing that, in Notebook 00."""
    for name in ("README.md", "00_data_acquisition_and_dictionary.ipynb"):
        p = ROOT / name
        m = BADGE.search(p.read_text(encoding="utf-8")) if p.exists() else None
        if m:
            return m.group(1), m.group(2), m.group(3)
    raise SystemExit("no Colab badge found in README.md or Notebook 00")


def files() -> list[Path]:
    out = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT).parts
        if p.is_file() and p.suffix in {".ipynb", ".md"} and not (
            SKIP & set(rel) or any(part.startswith(".venv") for part in rel)
        ):
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", help="owner/name on GitHub, e.g. PacktPublishing/New-Name")
    ap.add_argument("--folder", help="this chapter's folder in the repository, e.g. Chapter03")
    ap.add_argument("--branch", help="the branch the badges open (default: keep the current one)")
    ap.add_argument("--dry-run", action="store_true", help="list the files that would change")
    args = ap.parse_args()
    if args.repo and not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        ap.error("--repo must look like owner/name")
    for name in ("folder", "branch"):
        if getattr(args, name) and not re.fullmatch(r"[\w.-]+", getattr(args, name)):
            ap.error(f"--{name} may contain only letters, digits, '.', '_' and '-'")

    repo, branch, folder = current()
    new = (args.repo or repo, args.branch or branch, args.folder or folder)
    if new == (repo, branch, folder):
        print(f"already {repo}, branch {branch}, folder {folder}: nothing to do")
        return
    old_badge = f"colab.research.google.com/github/{repo}/blob/{branch}/{folder}/"
    new_badge = f"colab.research.google.com/github/{new[0]}/blob/{new[1]}/{new[2]}/"
    changed = 0
    for p in files():
        text = p.read_text(encoding="utf-8")
        out = text.replace(old_badge, new_badge).replace(f"https://github.com/{repo}", f"https://github.com/{new[0]}")
        if repo != new[0]:
            out = out.replace(f"`{repo}`", f"`{new[0]}`")
        if folder != new[2]:                          # the folder name as a whole word, anywhere
            out = re.sub(rf"(?<![\w-]){re.escape(folder)}(?![\w-])", new[2], out)
        if out != text:
            changed += 1
            print(("would change " if args.dry_run else "changed ") + str(p.relative_to(ROOT)))
            if not args.dry_run:
                p.write_text(out, encoding="utf-8")
    print(f"{repo}/{folder} (branch {branch}) -> {new[0]}/{new[2]} (branch {new[1]}): {changed} files")


if __name__ == "__main__":
    main()
