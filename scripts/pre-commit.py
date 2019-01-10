#!/usr/bin/env python3
# -*- mode: python -*-
from pathlib import Path
from subprocess import run, PIPE

here = Path(__file__).parent.absolute()

# get base path of directory from git hook or script folder
base = Path(str(here).split("obsplus")[0] + "/obsplus")

# run black
black_run = run(["black", str(base)], stdout=PIPE)

# clear notebook outputs (to be nice to git)
# keep_notebooks = {}  # long running notebooks that should not be cleared
# cmd = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace".split()
# notebooks = (base / "docs").rglob("*.ipynb")
# for notebook in notebooks:
#     is_checkpoint = "ipynb_checkpoint" in str(notebook)
#     if notebook.name in keep_notebooks or is_checkpoint:
#         continue
#     run(cmd + [str(notebook)], stdout=PIPE)

# re-stage all modified files (so the new versions get committed)
cmd = "git diff --name-only --cached | xargs -l git add".split(" ")
run(cmd)
