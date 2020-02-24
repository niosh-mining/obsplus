"""
Setup script for obsplus
"""
import glob
import sys
import versioneer
from collections import defaultdict
from os.path import join, exists, isdir

try:  # not running python 3, will raise an error later on
    from pathlib import Path
except ImportError:
    pass

from setuptools import setup

# define python versions
python_version = (3, 6)  # tuple of major, minor version requirement
python_version_str = str(python_version[0]) + "." + str(python_version[1])

# produce an error message if the python version is less than required
if sys.version_info < python_version:
    msg = "ObsPlus only runs on python version >= %s" % python_version_str
    raise Exception(msg)

# get path references
here = Path(__file__).absolute().parent
readme_path = here / "README.rst"
# get requirement paths
package_req_path = here / "requirements.txt"
test_req_path = here / "tests" / "requirements.txt"
doc_req_path = here / "docs" / "requirements.txt"


# --- utils


def find_packages(base_dir="."):
    """ setuptools.find_packages wasn't working so I rolled this """
    out = []
    for fi in glob.iglob(join(base_dir, "**", "*"), recursive=True):
        if isdir(fi) and exists(join(fi, "__init__.py")):
            out.append(fi)
    out.append(base_dir)
    return out


def get_package_data_files():
    """Return a list of datafiles to include in builds."""
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
    # add license/requirements so they get included in builds file
    out["."] = ["LICENSE", "requirements.txt"]
    return list(out.items())


def read_requirements(path, skip_if_missing=False):
    """ Read a requirements.txt file, return a list. """
    path = Path(path)
    if not path.exists() and skip_if_missing:
        return []
    with Path(path).open("r") as fi:
        lines = fi.readlines()
    # remove any line comments
    return [x for x in lines if not x.startswith("#")]


def load_file(path):
    """ Load a file into memory. """
    with Path(path).open() as w:
        contents = w.read()
    return contents


# --- get sub-packages


requires = read_requirements(package_req_path)
tests_require = read_requirements(test_req_path, skip_if_missing=True)
docs_require = read_requirements(doc_req_path, skip_if_missing=True)
dev_requires = tests_require + docs_require

ENTRY_POINTS = {
    "obsplus.datasets": [
        "bingham_test = obsplus.datasets.bingham_test",
        "crandall_test = obsplus.datasets.crandall_test",
        "ta_test = obsplus.datasets.ta_test",
        "default_test = obsplus.datasets.default_test",
    ]
}

df = get_package_data_files()

setup(
    name="obsplus",
    version=versioneer.get_version(),
    description="Some add-ons to obspy",
    long_description=load_file(readme_path),
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
    tests_require=tests_require,
    extras_require={"dev": dev_requires},
    python_requires=">=%s" % python_version_str,
    cmdclass=versioneer.get_cmdclass(),
)
