"""
Module for loading, (and downloading) data sets
"""
import abc
import copy
import json
import shutil
import tempfile
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType as MapProxy
from typing import Union, Optional
from warnings import warn

import obspy.clients.fdsn
import pkg_resources
from obspy.clients.fdsn import Client

from obsplus import Fetcher, WaveBank, EventBank
from obsplus.constants import DATA_TYPES
from obsplus.events.utils import get_event_client
from obsplus.exceptions import (
    FileHashChangedError,
    MissingDataFileError,
    DataVersionError,
)
from obsplus.stations.utils import get_station_client
from obsplus.utils import md5_directory
from obsplus.waveforms.utils import get_waveform_client

base_path = Path(__file__).parent


class DataSet(abc.ABC):
    """
    Class for downloading and serving datasets.

    By default the data will be downloaded to obsplus' datasets module
    but this can be changed using the base_path argument. All data will
    be saved in base_path / name.

    Parameters
    ----------
    base_path
        The base directory for the dataset.
    """

    _entry_points = {}
    datasets = {}
    base_path = base_path
    data_loaded = False
    # variables for hashing datafiles and versioning
    _version_path = ".version"
    _hash_path = "md5_hash.json"
    _hash_excludes = ("readme.txt", _version_path, _hash_path)

    # generic functions for loading data (WaveBank, event, stations)
    _load_funcs = MapProxy(
        dict(
            waveform=get_waveform_client,
            event=get_event_client,
            station=get_station_client,
        )
    )
    # flags to determine if data should be loaded into memory
    _load_waveforms = False
    _load_stations = True
    _load_events = True
    # cache for instantiated datasets
    _loaded_datasets = {}

    def __init_subclass__(cls, **kwargs):
        """ Register subclasses of datasets. """
        assert isinstance(cls.name, str), "name must be a string"
        cls._validate_version_str(cls.version)
        # Register the subclass as a dataset.
        DataSet.datasets[cls.name.lower()] = cls

    # --- logic for loading and caching data

    def __init__(self, base_path=None):
        """ download and load data into memory. """
        if base_path:  # overwrite main base path (mainly for testing)
            self.base_path = base_path
        # create the dataset's base directory
        self.path.mkdir(exist_ok=True, parents=True)
        # Make sure the version of the dataset is okay
        version_ok = self.check_version()
        # iterate each kind of data and download if needed
        downloaded = False
        for what in DATA_TYPES:
            needs_str = f"{what}s_need_downloading"
            if getattr(self, needs_str) or (not version_ok):
                # this is the first type of data to be downloaded, run fist hook
                if not downloaded:
                    self.pre_download_hook()
                downloaded = True
                # download data, test termination criteria
                print(f"downloading {what} data for {self.name} dataset ...")
                getattr(self, "download_" + what + "s")()
                assert not getattr(self, needs_str), f"Download {what} failed"
                print(f"finished downloading {what} data for {self.name}")
                self._write_readme()  # make sure readme has been written
        # some data were downloaded, call post download hook
        if downloaded:
            self.check_files()
            self.post_download_hook()
            # write a new version file
            self.write_version()
        self.data_loaded = True
        # cache loaded dataset
        if not base_path and self.name not in self._loaded_datasets:
            self._loaded_datasets[self.name] = self.copy(deep=True)

    def _load(self, what, path):
        try:
            client = self._load_funcs[what](path)
        except TypeError:
            warn(f"failed to load {what} from {path}, returning None")
            return None
        # load data into memory (eg load event bank contents into catalog)
        if getattr(self, f"_load_{what}s"):
            return getattr(client, f"get_{what}s")()
        else:
            return client

    def copy(self, deep=True):
        """
        Return a copy of the dataset.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)

    def copy_to(self, destination: Optional[Union[str, Path]] = None):
        """
        Copy the dataset to a destination.

        Parameters
        ----------
        destination
            The destination to copy the dataset. It will be created if it
            doesnt exist. If None is provided use tmpfile to create a temporary
            directory.

        Returns
        -------
        A new dataset object which refers to the copied files.
        """
        return copy_dataset(self, destination)

    def get_fetcher(self) -> Fetcher:
        """
        Return a Fetcher from the data.
        """
        assert self.data_loaded, "data have not been loaded into memory"
        # get events/waveforms/stations and put into dict for the Fetcher
        fetch_kwargs = {
            "waveforms": self.waveform_client,
            "events": self.event_client,
            "stations": self.station_client,
        }
        return Fetcher(**fetch_kwargs)

    __call__ = get_fetcher

    def _write_readme(self, filename="readme.txt"):
        """ Writes the classes docstring to a file. """
        path = self.path / filename
        if not path.exists():
            with path.open("w") as fi:
                fi.write(str(self.__doc__))

    @classmethod
    def load_dataset(cls, name: str) -> "DataSet":
        """
        Get a loaded dataset.

        Will ensure all files are downloaded and the appropriate data are
        loaded into memory.

        Parameters
        ----------
        name
            The name of the dataset to load.
        """
        name = name.lower()
        if name not in cls.datasets:
            # try to load it from entry points
            cls._load_dataset_entry_points()
            if name in cls._entry_points:
                cls._entry_points[name].load()
                return load_dataset(name)
            msg = f"{name} is not in the known datasets {list(cls.datasets)}"
            raise ValueError(msg)
        if name in cls._loaded_datasets:
            return cls._loaded_datasets[name].copy()
        else:
            return cls.datasets[name]()

    @classmethod
    def _load_dataset_entry_points(cls):
        """ load and cache the dataset entry points. """
        if not cls._entry_points:
            for ep in pkg_resources.iter_entry_points("obsplus.datasets"):
                cls._entry_points[ep.name] = ep

    # --- prescribed Paths for data

    @property
    def path(self) -> Path:
        return self.base_path / self.name

    @property
    def waveform_path(self) -> Path:
        return self.path / "waveforms"

    @property
    def event_path(self) -> Path:
        return self.path / "events"

    @property
    def station_path(self) -> Path:
        return self.path / "stations"

    # --- checks for if each type of data is downloaded

    @property
    def waveforms_need_downloading(self):
        """
        Returns True if waveform data need to be downloaded.
        """
        return not self.waveform_path.exists()

    @property
    def events_need_downloading(self):
        """
        Returns True if event data need to be downloaded.
        """
        return not self.event_path.exists()

    @property
    def stations_need_downloading(self):
        """
        Returns True if station data need to be downloaded.
        """
        return not self.station_path.exists()

    @property
    @lru_cache()
    def waveform_client(self) -> Optional[WaveBank]:
        """ A cached property for a waveform client """
        return self._load("waveform", self.waveform_path)

    @property
    @lru_cache()
    def event_client(self) -> Optional[EventBank]:
        """ A cached property for an event client """
        return self._load("event", self.event_path)

    @property
    @lru_cache()
    def station_client(self) -> Optional[obspy.Inventory]:
        """ A cached property for a station client """
        return self._load("station", self.station_path)

    @property
    @lru_cache()
    def _download_client(self):
        """
        Return an instance of the IRIS client, subclasses can override
        to use different clients.
        """
        return Client("IRIS")

    @_download_client.setter
    def _download_client(self, item):
        """ just allow this to be overwritten """
        self.__dict__["client"] = item

    def create_md5_hash(self, path=_hash_path, hidden=False) -> dict:
        """
        Create an md5 hash of all dataset's files to ensure dataset integrity.

        Keys are paths (relative to dataset base path) and values are md5
        hashes.

        Parameters
        ----------
        path
            The path to which the hash data is saved. If None dont save.
        hidden
            If True also include hidden files
        """
        out = md5_directory(self.path, exclude="readme.txt", hidden=hidden)
        if path is not None:
            # sort dict to mess less with git
            sort_dict = OrderedDict(sorted(out.items()))
            with (self.path / Path(path)).open("w") as fi:
                json.dump(sort_dict, fi)
        return out

    def check_files(self, check_hash=False):
        """
        Check that the files are all there and have the correct Hashes.

        Parameters
        ----------
        check_hash
            If True check the hash of the files.

        Returns
        -------

        """
        # TODO figure this out (data seem to have changed on IRIS' end)
        # If there is not a pre-existing hash file return
        hash_path = Path(self.path / self._hash_path)
        if not hash_path.exists():
            return
        # get old and new hash, and overlaps
        old_hash = json.load(hash_path.open())
        current_hash = md5_directory(self.path, exclude=self._hash_excludes)
        overlap = set(old_hash) & set(current_hash) - set(self._hash_excludes)
        # get any files with new hashes
        has_changed = {x for x in overlap if old_hash[x] != current_hash[x]}
        missing = (set(old_hash) - set(current_hash)) - set(self._hash_excludes)
        if has_changed and check_hash:
            msg = (
                f"The md5 hash for dataset {self.name} did not match the "
                f"expected values for the following files:\n{has_changed}"
            )
            raise FileHashChangedError(msg)
        if missing:
            msg = f"The following files are missing: \n{missing}"
            raise MissingDataFileError(msg)

    def check_version(self, path=_version_path):
        """
        Check the version of the dataset.

        Verifies the version string in the dataset class definintion matches
        the one saved on disk.

        Parameters
        ----------
        path
            Expected path of the version file.

        Raises
        ------
        DataVersionError
            If any version problems are discovered.

        Returns
        -------
        version_ok : bool
            True if the version matches what is expected.
        """
        path = self.path / path
        redownload_msg = (
            f"Delete the following files/directories to re-download the "
            f"dataset:\n {self.event_path}\n, {self.station_path}\n, "
            f"{self.waveform_path}\n, {path}\n"
        )
        try:
            version = self.read_data_version()
        except DataVersionError:  # The data version cannot be read from disk
            need_dl = (getattr(self, f"{x}s_need_downloading") for x in DATA_TYPES)
            if not any(need_dl):  # Something is a little weird
                warn("Version file is missing or corrupt. Attempting to re-download the dataset.")
            return False
        # Check the version number
        if version < self.version:
            msg = f"Dataset version is out of date: {version} < {self.version}. "
            raise DataVersionError(msg + redownload_msg)
        elif version > self.version:
            msg = f"Dataset version mismatch: {version} > {self.version}."
            msg = msg + " It may be necessary to reload the dataset."
            warn(msg + redownload_msg)
        return True  # All is well. Continue.

    def write_version(self):
        """ Write the version string to disk. """
        version_path = Path(self.path) / self._version_path
        with version_path.open("w") as fi:
            fi.write(self.version)

    def read_data_version(self):
        """
        Read the data version from disk.

        Raise a DataVersionError if not found.
        """
        version_path = Path(self.path) / self._version_path
        if not version_path.exists():
            raise DataVersionError(f"{version_path} does not exist!")
        with version_path.open("r") as fi:
            version_str = fi.read()
        self._validate_version_str(version_str)
        return version_str

    @staticmethod
    def _validate_version_str(version_str):
        """
        Check the version string is of the form x.y.z.

        If the version string is not valid raise DataVersionError.
        """
        is_str = isinstance(version_str, str)
        has_3 = len(version_str.split(".")) == 3
        if not (is_str and has_3):
            msg = f"version must be a string of the form x.y.z, not {version_str}"
            raise DataVersionError(msg)

    # --- Abstract properties subclasses should implement
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the dataset
        """

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """
        Dataset version. Should be a str of the form x.y.z
        """

    # --- Abstract methods subclasses should implement

    def download_events(self) -> None:
        """
        Method to ensure the events have been downloaded.

        Events should be written in an obspy-readable format to
        self.event_path. If not implemented this method will create an empty
        directory.
        """
        self.event_path.mkdir(exist_ok=True, parents=True)

    def download_waveforms(self) -> None:
        """
        Method to ensure waveforms have been downloaded.

        Waveforms should be written in an obspy-readable format to
        self.waveform_path.
        """
        self.waveform_path.mkdir(exist_ok=True, parents=True)

    @abc.abstractmethod
    def download_stations(self) -> None:
        """
        Method to ensure inventories have been downloaded.

        Station data should be written in an obspy-readable format to
        self.station_path. Since there is not yet a functional StationBank,
        this method must be implemented by subclass.
        """

    def pre_download_hook(self):
        """ Code to run before any downloads. """

    def post_download_hook(self):
        """ code to run after any downloads. """

    def __str__(self):
        return f"Dataset: {self.name}"

    def __repr__(self):
        return f"{str(self)} with description: {self.__doc__}"


load_dataset = DataSet.load_dataset


def copy_dataset(
    dataset: Union[str, DataSet], destination: Optional[Union[str, Path]] = None
) -> DataSet:
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
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)
    expected_path: Path = dataset.path
    assert expected_path.exists(), f"{expected_path} not yet downloaded"
    # make destination paths and copy
    if destination is None:
        dest_dir = Path(tempfile.mkdtemp())
    else:
        dest_dir = Path(destination)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / dataset.name
    shutil.copytree(str(expected_path), str(dest))
    # init new dataset of same class with updated base_path and return
    return dataset.__class__(base_path=dest.parent)
