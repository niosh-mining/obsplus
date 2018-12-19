"""
Module for loading, (and downloading) data sets
"""
import abc
import copy
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType as MapProxy
from typing import Union, Optional

import obspy.clients.fdsn
import pkg_resources
from obspy.clients.fdsn import Client

from obsplus import Fetcher, WaveBank, EventBank
from obsplus.events.utils import get_event_client
from obsplus.stations.utils import get_station_client
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
    name = None
    # cache for loaded objects
    event_client: Optional[EventBank] = None
    waveform_client: Optional[WaveBank] = None
    station_client: Optional[obspy.Inventory] = None
    data_loaded = False
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
        """ Register instances of datasets. """
        assert hasattr(cls, "name"), "must have a name"
        assert isinstance(cls.name, str), "name must be a string"
        DataSet.datasets[cls.name.lower()] = cls

    # --- logic for loading and caching data

    def __init__(self, base_path=None):
        """ download and load data into memory. """
        if base_path:  # overwrite main base path (mainly for testing)
            self.base_path = base_path
        self.path.mkdir(exist_ok=True, parents=True)
        for what in ["waveform", "event", "station"]:
            path = getattr(self, what + "_path")
            # the data have not yet been downloaded
            if not path.exists():
                # download data, ensure the expected paths are created
                print(f"downloading {what} data for {self.name} dataset ...")
                getattr(self, "download_" + what + "s")()
                assert path.exists(), f"after download {path} does not exist!"
                print(f"finished downloading {what} data for {self.name}")
                self._write_readme()  # make sure readme has been written
            # data are downloaded, but not yet loaded
            if what not in self.__dict__:
                setattr(self, what + "_client", self._load(what, path))
        self.data_loaded = True
        # cache loaded dataset
        if not base_path and self.name not in self._loaded_datasets:
            self._loaded_datasets[self.name] = self.copy(deep=True)

    def _load(self, what, path):
        client = self._load_funcs[what](path)
        # load data into memory (eg load event bank contents into events)
        if getattr(self, f"_load_{what}s"):
            return getattr(client, f"get_{what}s")()
        else:
            return client

    def copy(self, deep=True):
        """
        Return a copy of the dataset.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)

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

    def _write_readme(self):
        """ Writes the classes docstring to a file. """
        path = self.path / "readme.txt"
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
            The name of the dataset. Supported values are in the
            obsplus.datasets.dataload.DATASETS dict
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
        assert self.name is not None, "subclass must define name"
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
    None
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
