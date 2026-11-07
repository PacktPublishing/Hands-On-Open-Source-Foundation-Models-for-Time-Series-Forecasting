# Maintaining this repository

For the authors: how chapters go into this repository, how to check that a reader who clones it can run them, and
how to publish to [PacktPublishing/Hands-On-Open-Source-Foundation-Models-for-Time-Series-Forecasting](https://github.com/PacktPublishing/Hands-On-Open-Source-Foundation-Models-for-Time-Series-Forecasting).

## Adding a chapter

1. Put the chapter in its own top-level folder named `ChapterNN_<Topic>` (two-digit number, underscores, no
   spaces), for example `Chapter05_Fine_Tuning`. Everything the chapter needs lives inside that folder: notebooks,
   `src/`, requirement files, data, stored results, tests and a README. Do not import code from another chapter;
   copy it if two chapters need it.
2. Give every notebook an *Open in Colab* badge in its first cell and a setup cell that clones this repository on
   Colab and moves into the chapter's folder. Chapter 3's notebooks show the pattern: the badge opens
   `.../blob/main/<folder>/<notebook>.ipynb` and the setup cell has `CHAPTER = "<folder>"`. Chapter 3's
   `scripts/set_repo_path.py` rewrites every badge, clone URL and `CHAPTER` line in one go if the folder, the
   repository or the branch is renamed:

   ```bash
   cd Chapter03_Zero_Shot_Forecasting
   python scripts/set_repo_path.py --folder Chapter03_New_Name --dry-run   # list the files it would change
   python scripts/set_repo_path.py --folder Chapter03_New_Name             # rewrite them
   python scripts/set_repo_path.py --repo PacktPublishing/Other-Name       # another repository
   ```

3. Give the chapter its own `.gitignore` for anything its notebooks write that should not be committed (quick-check
   outputs, downloads). The `.gitignore` at the root already ignores environments, caches, `.DS_Store`, `.env` files
   and every chapter's `data/raw/` and `data/rebuild/`.
4. Add a row for the chapter to the table in the root [README.md](README.md).
5. Keep every file under 50 MB (GitHub warns at 50 MB and refuses 100 MB). Chapter 3 is about 130 MB in 653 files;
   its largest file is a 7 MB notebook. If a later chapter is much heavier, move bulky stored results to a GitHub
   release and download them in the notebook.
6. Never commit tokens or passwords. Notebooks that need a key read it from an environment variable.

## Checking a chapter as a reader would

Run this before every push. It clones the repository into a fresh folder, so it sees exactly what a reader gets
(files ignored by git are missing, nothing from your own environments leaks in). Replace the URL with the path of
this folder to test before pushing (`git clone /path/to/Packt_GitHub_Repo /tmp/fresh` clones your last commit).

```bash
git clone https://github.com/PacktPublishing/Hands-On-Open-Source-Foundation-Models-for-Time-Series-Forecasting.git /tmp/fresh
cd /tmp/fresh/Chapter03_Zero_Shot_Forecasting

# one environment per notebook group, as listed in the chapter README
uv venv .venv-dataprep -p 3.12 && uv pip install -p .venv-dataprep -r requirements-dataprep.txt
.venv-dataprep/bin/python -m pytest -q
.venv-dataprep/bin/python ../tools/run_notebooks.py 0[0-4]_*.ipynb 1[12]_*.ipynb bonus/B1[12]_*.ipynb

uv venv .venv-core -p 3.12 && uv pip install -p .venv-core -r requirements/env-core.txt
.venv-core/bin/python ../tools/run_notebooks.py 05a_*.ipynb 05b_*.ipynb bonus/B05a_*.ipynb bonus/B05b_*.ipynb
# ... and so on for every environment in the chapter README

git status --short     # what the runs changed; quick-check outputs are ignored, so this is usually short
```

`tools/run_notebooks.py` prints `passed` or `failed` and the run time for each notebook, and the first error of a
failed one; `--out DIR` keeps the executed notebooks so you can read the error in context.

On Colab, a reader's path is: badge, then *Runtime > Run all*; the setup cell clones the repository and installs
the notebook's environment from what is on GitHub, so a chapter's badges work as soon as it is pushed (the
repository is public). To test changes on Colab before pushing them, use a copy of the chapter folder on Google
Drive (set `DRIVE_COPY` in the setup cell).

## Publishing

Upload with git, not the GitHub web page: the web page takes at most 100 files at a time and does not read
`.gitignore`.

Packt's repository is public and starts with one commit of their own (a two-line README and their MIT LICENSE,
7 September 2026). This folder already carries Packt's LICENSE unchanged and replaces their README with the book's.
The first time, link this folder to that history and push:

```bash
cd Packt_GitHub_Repo
git init -b main
git remote add origin https://github.com/PacktPublishing/Hands-On-Open-Source-Foundation-Models-for-Time-Series-Forecasting.git
git fetch origin
git reset origin/main      # take Packt's commit as the starting point; your files stay as they are
git add .
git status --short         # README.md modified, everything else new; no .venv, data/raw, _smoke or .DS_Store
git commit -m "Add Chapter 3: Zero-Shot Forecasting in Practice"
git push -u origin main
```

GitHub asks you to sign in on the first push; use your GitHub account (it needs write access to the repository,
which Packt grants) and a personal access token as the password, or sign in through GitHub Desktop or the GitHub CLI
(`gh auth login`).

After that, each new or updated chapter is `git add ChapterNN_<Topic> && git commit -m "..." && git push`.
Every commit of a changed notebook adds its full size to the repository's history, so push a chapter once it has
been checked, not after every small edit.
