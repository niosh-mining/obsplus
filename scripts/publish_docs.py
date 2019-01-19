"""
Script to make the documentation and publish to gh-pages. This should
be done once for each release.
"""
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from subprocess import run, PIPE

from clean_docs import main as clean_docs
from make_docs import main as make_docs

import obsplus

version = obsplus.__version__


@contextmanager
def change_directory(new_path):
    here = Path(".")
    os.chdir(new_path)
    yield
    os.chdir(here)


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    tmp = Path(tempfile.mkdtemp())
    html_path = (base / "docs" / "_build" / "html").absolute()
    obsplus_url = "https://github.com/niosh-mining/obsplus"
    # make the docs
    make_docs()
    assert html_path.exists()
    # clone a copy of obplus to tempfile and cd there
    run(f"git clone {obsplus_url}", cwd=tmp, shell=True)
    with change_directory(tmp / "obsplus"):
        # checkout gh-pages
        run("git checkout gh-pages", shell=True, check=True)
        # reset to the first commit in gh-pages branch
        cmd = "git reset --hard `git rev-list --max-parents=0 HEAD | tail -n 1`"
        run(cmd, shell=True, stdout=PIPE, stderr=PIPE, check=True)
        # copy all contents of html
        run(f"cp -R {html_path}/* ./", shell=True, check=True)
        # make a commit
        run("git add -A", shell=True)
        run(f'git commit -m "{version} docs"', shell=True, check=True)
        run("git push -f origin gh-pages", shell=True, stdout=PIPE, check=True)
    # clean up original repo
    clean_docs()
