#!/usr/bin/env python
"""
Pre-commit hook for ensuring files have been black formatted.
"""
from pathlib import Path
from subprocess import run, PIPE

here = Path(__file__).parent.absolute()

# get base path of directory from git hook or script folder
base = Path(str(here).split("obsplus")[0] + "/obsplus")

# run black
black_run = run(["black", str(base), "--check"], stdout=PIPE)
if black_run.returncode != 0:
    raise ValueError("You must run black formatter before committing!")
