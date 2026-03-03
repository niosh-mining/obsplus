"""
A script to publush a new version of Obsplus to pypi.
"""
import shutil
from pathlib import Path
from subprocess import run


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    # clear out old build/dist paths
    build_path = base / "build"
    dist_path = base / "dist"
    for path in [build_path, dist_path]:
        if path.is_dir():
            shutil.rmtree(path)

    run("python setup.py sdist bdist_wheel", shell=True, cwd=base)
    run("twine upload dist/*", shell=True, cwd=base)
    test_pypi_url = "--repository-url https://test.pypi.org/legacy/"
    # run("twine upload {test_pypi_url}  dist/*", shell=True, cwd=base)
