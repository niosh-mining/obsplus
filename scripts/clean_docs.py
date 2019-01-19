"""
Script to remove all notebook output from docs, as well as old sphinx outputs.
"""
import shutil
from pathlib import Path
from subprocess import run


def main():
    doc_path = Path(__file__).absolute().parent.parent / "docs"
    # make api documentation
    cmd = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace"
    api_path = doc_path / "api"
    # clear API documentation
    if api_path.exists():
        shutil.rmtree(api_path)
    # execute all the notebooks
    for note_book_path in doc_path.rglob("*.ipynb"):
        result = run(cmd + f" {note_book_path}", shell=True)
        if result.returncode != 0:
            msg = f"failed to execute {note_book_path}!"
            raise RuntimeError(msg)


if __name__ == "__main__":
    main()
