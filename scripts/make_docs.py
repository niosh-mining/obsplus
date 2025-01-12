"""
Script to re-make the html docs and publish to gh-pages.
"""

from pathlib import Path
from subprocess import run

import typer
from clean_docs import clean_docs

# Path to top-level sphinx
DOC_PATH = Path(__file__).absolute().parent.parent / "docs"


def make_docs(doc_path=DOC_PATH, timeout=3000) -> str:
    """
    Make the obsplus documentation.

    Parameters
    ----------
    doc_path
        The path to the top-level sphinx directory.
    timeout
        Time in seconds allowed for notebook to build.

    Returns
    -------
    Path to created html directory.

    """
    doc_path = Path(doc_path)
    # clean out all the old docs
    clean_docs()
    # execute all the notebooks
    cmd = (
        f"jupyter nbconvert --to notebook --execute --inplace "
        f"--ExecutePreprocessor.timeout={timeout}"
    )
    for note_book_path in doc_path.rglob("*.ipynb"):
        # skip all ipython notebooks
        if "ipynb_checkpoints" in str(note_book_path):
            continue
        result = run(cmd + f" {note_book_path}", shell=True)
        if result.returncode != 0:
            raise RuntimeError(f"failed to run {note_book_path}")
    # run auto api-doc
    run("sphinx-apidoc ../src/obsplus -e -M -o api", cwd=doc_path, shell=True)
    run("make html", cwd=doc_path, shell=True, check=True)
    # ensure html directory was created, return path to it.
    expected_path: Path = doc_path / "_build" / "html"
    assert expected_path.is_dir(), f"{expected_path} does not exist!"
    return str(expected_path)


if __name__ == "__main__":
    typer.run(make_docs)
