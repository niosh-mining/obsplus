"""
Use Black to check the codebase and tests
"""
import subprocess
from pathlib import Path

import pytest

# skip if black is not installed
pytest.importorskip("black")

# get the path to all python files that should be formatted
base = Path(__file__).absolute().parent.parent
src_files = list((base / "obsplus").rglob("*.py"))
test_files = list((base / "tests").rglob("*.py"))
py_files = src_files + test_files


@pytest.mark.parametrize("py_file", py_files)
def test_format(py_file):
    """ Run the black formatter check on specified file. """
    cmd = ["black", "--check", "--diff", "--quiet", str(py_file)]
    run = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    if run.returncode != 0:
        pytest.fail(run.stdout)
