"""
A script to publush a new version of Obsplus to pypi.
"""
from pathlib import Path
from subprocess import run


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    run('python setup.py sdist bdist_wheel', shell=True, cwd=base)
    run('twine upload dist/*', shell=True, cwd=base)
