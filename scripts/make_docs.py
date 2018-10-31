"""
Script to re-make the html docs.
"""
import shutil
from pathlib import Path
from subprocess import run


def main():
    doc_path = Path(__file__).absolute().parent.parent / "docs"
    # make api documentation
    api_path = doc_path / "api"
    if api_path.exists():
        shutil.rmtree(api_path)
    # run auto api-doc
    run(f" sphinx-apidoc ../obsplus -o api", cwd=doc_path, shell=True)
    run(f"make html", cwd=doc_path, shell=True)


if __name__ == "__main__":
    main()
