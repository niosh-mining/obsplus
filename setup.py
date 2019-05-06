"""
Setup script for obsplus
"""
import glob
import os
import shutil
import stat
import sys
from collections import defaultdict
from os.path import join, exists, isdir

try:  # not running python 3, will raise an error later on
    from pathlib import Path
except ImportError:
    pass

from setuptools import setup
from setuptools.command.develop import develop

# define python versions

python_version = (3, 6)  # tuple of major, minor version requirement
python_version_str = str(python_version[0]) + "." + str(python_version[1])

# produce an error message if the python version is less than required
if sys.version_info < python_version:
    msg = "ObsPlus only runs on python version >= %s" % python_version_str
    raise Exception(msg)

# get path references
here = Path(__file__).absolute().parent
version_file = here / "obsplus" / "version.py"

# --- get version
with version_file.open() as fi:
    content = fi.read().split("=")[-1].strip()
    __version__ = content.replace('"', "").replace("'", "")


# --- get readme
with open("README.rst") as readme_file:
    readme = readme_file.read()


# --- get sub-packages
def find_packages(base_dir="."):
    """ setuptools.find_packages wasn't working so I rolled this """
    out = []
    for fi in glob.iglob(join(base_dir, "**", "*"), recursive=True):
        if isdir(fi) and exists(join(fi, "__init__.py")):
            out.append(fi)
    out.append(base_dir)
    return out


def get_package_data_files():
    """ Gets data """
    data = Path("obsplus") / "datasets"
    out = defaultdict(list)
    # get list of datasets
    datasets = [x for x in data.glob("*") if x.is_dir()]
    for dataset in datasets:
        for ifile in dataset.glob("*"):
            if ifile.name.endswith("py") or ifile.name.endswith("pyc"):
                continue
            if ifile.is_dir():
                continue
            out[str(ifile.parent)].append(str(ifile))
    return list(out.items())


# --- requirements paths


def read_requirements(path):
    """ Read a requirements.txt file, return a list. """
    with Path(path).open("r") as fi:
        return fi.readlines()


package_req_path = here / "requirements.txt"
test_req_path = here / "tests" / "requirements.txt"
doc_req_path = here / "docs" / "requirements.txt"


class SetupDev(develop):
    """
    Install the obsplus git hook when development mode is run.
    These hooks run the black formatter on the code base and clear
    notebook outputs.
    """

    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)
        # get path to get repo, do nothing if it doesnt exist
        gitpath = Path(".git")
        if not gitpath.exists():
            return
        # copy hooks
        hook_script = Path("scripts") / "pre-commit.py"
        assert hook_script.exists()
        # copy to git directory
        out_path = gitpath / "hooks" / "pre-commit"
        shutil.copy(hook_script, out_path)
        # make sure script is executable
        st = os.stat(str(out_path))
        os.chmod(out_path, st.st_mode | stat.S_IEXEC)


ENTRY_POINTS = {
    "obsplus.datasets": [
        "bingham = obsplus.datasets.bingham",
        "crandall = obsplus.datasets.crandall",
        "kemmerer = obsplus.datasets.kemmerer",
        "ta = obsplus.datasets.ta",
    ]
}

df = get_package_data_files()

setup(
    name="obsplus",
    version=__version__,
    description="Some add-ons to obspy",
    long_description=readme,
    author="Derrick Chambers",
    author_email="djachambeador@gmail.com",
    url="https://github.com/niosh-mining/obsplus",
    packages=find_packages("obsplus"),
    package_dir={"obsplus": "obsplus"},
    entry_points=ENTRY_POINTS,
    include_package_data=True,
    data_files=get_package_data_files(),
    license="GNU Lesser General Public License v3.0 or later (LGPLv3.0+)",
    zip_safe=False,
    keywords="obsplus",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    test_suite="tests",
    install_requires=read_requirements(package_req_path),
    tests_require=read_requirements(test_req_path),
    setup_requires=["pytest-runner>=2.0"],
    extras_require={"docs": read_requirements(doc_req_path)},
    cmdclass={"develop": SetupDev},
    python_requires=">=%s" % python_version_str,
)
