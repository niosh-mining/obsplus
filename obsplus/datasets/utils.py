"""
Simple utility for DataSet.
"""
import shutil
import tempfile
import textwrap
import os
from pathlib import Path
from typing import Optional, Union

import obsplus
from obsplus.constants import OPSDATA_PATH


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

    1. Look for an environmental name OPSDATA_PATH, if defined return it.
    2. Get the opsdata_path variable from obsplus.constants and return it.

    Returns
    -------
    A path to the opsdata directory.
    """
    if opsdata_path is None:
        opsdata_path = os.getenv("OPSDATA_PATH", OPSDATA_PATH)
    _create_opsdata(opsdata_path)
    return opsdata_path


def copy_dataset(
    dataset: Union[str, "DataSet"], destination: Optional[Union[str, Path]] = None
) -> "DataSet":
    """
    Copy a dataset to a destination.

    Parameters
    ----------
    dataset
        The name of the dataset or a DataSet object. Supported str values are
        in the obsplus.datasets.dataload.DATASETS dict.
    destination
        The destination to copy the dataset. It will be created if it
        doesnt exist. If None is provided use tmpfile to create a temporary
        directory.

    Returns
    -------
    A new dataset object which refers to the copied files.
    """
    dataset = obsplus.load_dataset(dataset)
    expected_path: Path = dataset.download_path
    assert expected_path.exists(), f"{expected_path} not yet downloaded"
    # make destination paths and copy
    if destination is None:  # use a temp directory if none specified
        dest_dir = Path(tempfile.mkdtemp())
    else:
        dest_dir = Path(destination)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / dataset.name
    shutil.copytree(str(expected_path), str(dest))
    # init new dataset of same class with updated base_path and return
    return dataset.__class__(base_path=dest.parent)
