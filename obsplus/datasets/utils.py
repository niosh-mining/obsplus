"""
Simple utility for DataSet.
"""
import textwrap
from pathlib import Path
from typing import Optional


def _create_opsdata(opsdata_path: Path):
    """ Create the directory to store obsplus datasets, add readme. """
    opsdata_path = Path(opsdata_path)
    # bail out early if the directory already exists
    if opsdata_path.exists():
        return
    # else create the directory and add a readme.
    opsdata_path.mkdir(parents=True, exist_ok=True)
    readme_path = opsdata_path / "README.txt"
    msg = textwrap.dedent(
        """
    This directory contains the data sets curated by the obsplus python
    package (github.com/niosh-mining/obsplus). 
    
    Each sub-directory contains a single data set and the data set's name is
    the name of the directory. You can load the dataset using the
    obsplus.load_dataset function and passing the name of the data set as a
    string.
    """
    )
    with readme_path.open("w") as fi:
        fi.write(msg)


def get_opsdata_path(opsdata_path: Optional[Path] = None) -> Path:
    """
    Simple script to get the location where datasets are stored.

    Uses the following priorities:

    1. Look for an environmental name opsdata_path, if defined return it.
    2. Get the opsdata_path variable from obsplus.constants and return it.

    Returns
    -------
    A path to the opsdata directory.
    """
    if opsdata_path is None:
        pass
