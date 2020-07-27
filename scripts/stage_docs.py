"""
Script to make the documentation and publish to gh-pages. This should
be done once for each release.
"""
import os
import shutil
from contextlib import contextmanager, suppress
from pathlib import Path
from subprocess import run, PIPE

import jinja2
import typer

from make_docs import make_docs

import obsplus

VERSION = obsplus.__version__

INDEX_TEMPLATE = """
<head>
      <title>ObsPlus Docs</title>
</head>
<body>
      <h1>ObsPlus Documentation</h1>
      <p><a href="versions/latest/">latest</a></p>
      {% for version in release_versions %}
      <p><a href="versions/{{ version }}/">{{ version }}</a></p>
      {% endfor %}
</body>
"""


@contextmanager
def change_directory(new_path):
    """ Temporarily change directories. """
    here = Path(".")
    os.chdir(new_path)
    yield
    os.chdir(here)


def _make_gh_pages_repo(repo_path, target_path):
    """Make the github pages repo."""
    # copy the current directory (obsplus repo) to target, delete if exists
    target = Path(target_path)
    repo = Path(repo_path)
    if target.is_dir():
        shutil.rmtree(target)
    shutil.copytree(repo, target)
    # checkout gh-pages, return path
    run("git reset --hard", cwd=target_path, shell=True)
    run("git branch -D gh-pages", cwd=target_path, shell=True)
    run("git fetch origin", cwd=target_path, shell=True)
    run("git checkout -t origin/gh-pages", cwd=target_path, shell=True)
    run("git reset --hard", cwd=target_path, shell=True)
    return target_path


def _get_release_versions(versions):
    """Get release versions from a list"""
    out = []
    for version_str in versions:
        con1 = "dirty" not in str(version_str)
        con2 = len(str(version_str).split(".")) == 3
        if con1 and con2:
            out.append(version_str)
    return list(reversed(sorted(out)))


def _build_index(pages_path, remove_dirty=False):
    """Builds the index.html at the top of pages."""

    def remove_dirty_dirs(kwargs):
        """Remove the doc directories build from dirty (non-release) commits."""
        for dirty_version in kwargs["non_release_versions"]:
            with suppress((NotADirectoryError, FileNotFoundError)):
                shutil.rmtree(pages_path / "versions" / dirty_version)
        kwargs["non_release_versions"].clear()

    def _create_version_dict(pages_path):
        """Get a dict of releaes/non-release version numbers."""
        versions_path = pages_path / "versions"
        versions = [x.name for x in versions_path.glob("*")]
        kwargs = dict(
            release_versions=_get_release_versions(versions),
            non_release_versions=[x for x in versions if "dirty" in str(x)],
        )
        return kwargs

    def _create_latest(pages_path, kwargs):
        """ Create the 'latest' version directory. """
        latest_version = VERSION
        latest_path = pages_path / "versions" / "latest"
        if latest_path.is_dir():
            shutil.rmtree(latest_path)
        from_path = pages_path / "versions" / latest_version
        shutil.copytree(from_path, latest_path)

    def _render_html(kwargs):
        """Render and write the index.html file."""
        html = jinja2.Template(INDEX_TEMPLATE).render(**kwargs)
        with index_path.open("w") as fi:
            fi.write(html)

    # ensure index is empty
    index_path = pages_path / "index.html"
    if index_path.exists():
        index_path.unlink()
    # get a list of versions
    kwargs = _create_version_dict(pages_path)
    # create latest
    _create_latest(pages_path, kwargs)
    # remove all dirty doc branches
    if remove_dirty:
        remove_dirty_dirs(kwargs)
    # render html, save to disk
    _render_html(kwargs)
    assert index_path.exists(), f"Failed to create {index_path}!"


def _commit_new_docs(pages_path):
    """Commit the new docs, overwrite second commit."""
    # remove precommit hooks
    cmd = "pre-commit uninstall"
    run(cmd, shell=True, stdout=PIPE, stderr=PIPE, check=True, cwd=pages_path)
    # reset to the first commit in gh-pages branch
    cmd = "git reset --soft `git rev-list --max-parents=0 HEAD | tail -n 1`"
    run(cmd, shell=True, stdout=PIPE, stderr=PIPE, check=True, cwd=pages_path)
    # make a commit
    run("git add -A", shell=True, check=True, cwd=pages_path)
    cmd = f'git commit -m "{VERSION} docs"'
    run(cmd, shell=True, check=True, cwd=pages_path)


def stage_docs(build_path=None, pages_path=None, remove_dirty: bool = False) -> str:
    """
    Stage ObsPlus' docs in copy of repo checking out branch GH-pages.

    Parameters
    ----------
    build_path
        Path to built HTML directory. If None call create
    pages_path
        Path to this repo checked out in GH-pages. If not provided simply
        create a new directory called "ops_docs" on the same level as the
        obsplus directory.
    remove_dirty
        If True, remove all docs built on dirty commits.

    Returns
    -------
        Path to staged GH-pages directory.
    """
    # get paths
    base = Path(__file__).absolute().parent.parent
    build_path = Path(make_docs()).parent if build_path is None else build_path
    pages_path = Path(pages_path or base.parent / "obsplus_documentation")
    _make_gh_pages_repo(base, pages_path)
    # copy build html directory
    expected_html = build_path / "html"
    new_path = pages_path / "versions" / VERSION
    if new_path.exists():
        shutil.rmtree(new_path)
    shutil.copytree(expected_html, new_path)
    # build new doc index
    _build_index(pages_path, remove_dirty)
    # commit
    _commit_new_docs(pages_path)
    return str(pages_path)


if __name__ == "__main__":
    typer.run(stage_docs)
