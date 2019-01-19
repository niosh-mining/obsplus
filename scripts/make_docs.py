"""
Script to re-make the html docs.

This should be run only on the docs branch. Other branches should run
the clean_docs script to ensure no notebook output is committed/merged into
master.
"""
from pathlib import Path
from subprocess import run

from clean_docs import main as clean_docs


def main():
    doc_path = Path(__file__).absolute().parent.parent / "docs"
    # clean out all the old docs
    clean_docs()
    # execute all the notebooks
    cmd = "jupyter nbconvert --to notebook --execute"
    for note_book_path in doc_path.rglob("*.ipynb"):
        result = run(cmd + f" {note_book_path}", shell=True)
        if result.returncode != 0:
            raise RuntimeError(f"failed to run {note_book_path}")
    # run auto api-doc
    run(f" sphinx-apidoc ../obsplus -o api", cwd=doc_path, shell=True)
    run(f"make html", cwd=doc_path, shell=True)


if __name__ == "__main__":
    main()
