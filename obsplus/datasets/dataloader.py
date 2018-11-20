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

import obspy
import obspy.clients.fdsn
from obspy import UTCDateTime as UTC
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.mass_downloader import (
    RectangularDomain,
    Restrictions,
    MassDownloader,
    CircularDomain,
)
from obspy.geodetics import kilometers2degrees

from obsplus import Fetcher, WaveBank, events_to_df, EventBank
from obsplus.events.utils import catalog_to_directory
from obsplus.events.utils import get_event_client
from obsplus.stations.utils import get_station_client
from obsplus.utils import make_time_chunks
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
            msg = f"{name} is not in the known datasets {list(cls.datasets)}"
            raise ValueError(msg)
        if name in cls._loaded_datasets:
            return cls._loaded_datasets[name].copy()
        else:
            return cls.datasets[name]()

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


class TA(DataSet):
    """
    A small dataset from the TA with low sampling rate channels.

    Contains about 10 days of data from two stations: M11A and M14A.
    """

    name = "TA"

    level = "response"
    starttime = obspy.UTCDateTime("2007-02-15")
    endtime = obspy.UTCDateTime("2007-02-25")

    bulk = [
        ("TA", "M11A", "*", "VH*", starttime, endtime),
        ("TA", "M14A", "*", "VH*", starttime, endtime),
    ]

    def download_stations(self):
        inv_path = self.station_path / "stations.xml"
        inv_path.parent.mkdir(exist_ok=True, parents=True)
        inv = self._download_client.get_stations_bulk(self.bulk, level=self.level)
        inv.write(str(inv_path), "stationxml")

    def download_waveforms(self):
        st = self._download_client.get_waveforms_bulk(self.bulk)
        self.build_archive(st)
        # update the index
        WaveBank(self.waveform_path).update_index()

    def build_archive(self, st, starttime=None, endtime=None):
        """
        Build the archive (dont use mass downloader for a little variety)
        """
        starttime = starttime or self.starttime
        endtime = endtime or self.endtime

        for utc1, utc2 in make_time_chunks(starttime, endtime, duration=3600):
            stt = st.copy()
            stt.trim(starttime=utc1, endtime=utc2)
            channels = {x.stats.channel for x in st}
            for channel in channels:
                ts = stt.select(channel=channel)
                if not len(ts):
                    continue
                stats = ts[0].stats
                net, sta = stats.network, stats.station
                loc, cha = stats.location, stats.channel
                utc_str = str(utc1).split(".")[0].replace(":", "-")
                end = utc_str + ".mseed"
                fpath = self.waveform_path / net / sta / loc / cha / end
                fpath.parent.mkdir(parents=True, exist_ok=True)
                ts.write(str(fpath), "mseed")


class Kemmerer(DataSet):
    """
    Another small TA dataset based on stations near Kemmerer Wyoming.

    Contains stations M17A and M18A as well as some recorded surface blasts
    from a nearby surface coal mine.
    """

    name = "kemmerer"

    bulk = [("TA", "M17A", "*", "BH?"), ("AT", "M18A", "*", "BH?")]

    def download_events(self):
        """ Simply copy events from base directory. """
        cat_path = base_path / self.name / "events.xml"
        assert cat_path.exists(), "this should ship with obsplus"
        cat = obspy.read_events(str(cat_path))
        catalog_to_directory(cat, self.event_path)
        # update index
        EventBank(self.event_path).update_index()

    def _download_kemmerer(self):
        """ downloads both stations and """
        for station in ["M17A", "M18A"]:
            domain = RectangularDomain(
                minlatitude=40.0,
                maxlatitude=43.0,
                minlongitude=-111.0,
                maxlongitude=-110.0,
            )
            restrictions = Restrictions(
                starttime=obspy.UTCDateTime("2009-04-01T00:00:00"),
                endtime=obspy.UTCDateTime("2009-04-04T00:00:00"),
                chunklength_in_sec=3600,
                network="TA",
                channel="BH?",
                station=station,
                reject_channels_with_gaps=False,
                minimum_length=0.0,
                minimum_interstation_distance_in_m=10.0,
            )
            MassDownloader(providers=[self._download_client]).download(
                domain,
                restrictions,
                mseed_storage=str(self.waveform_path),
                stationxml_storage=str(self.station_path),
            )

    download_stations = _download_kemmerer
    download_waveforms = _download_kemmerer


class Crandall(DataSet):
    """
    Seismic data recorded by regional stations associated with the Crandall
    Canyon Mine Collapse (goo.gl/2ezyZD). Six mine workers and three rescue
    workers lost their lives in the disaster.

    This dataset is of interest to mine seismology researchers because better
    understanding of the conditions and sources involved will help prevent
    future disasters.

    I have personally modified and added some phase picks to the events.
    Without closer examination, this dataset should be used for testing
    and demonstration purposes only.
    """

    name = "crandall"
    # Days of interest. The collapse occurred on Aug 6th
    starttime = obspy.UTCDateTime("2007-08-06")
    endtime = obspy.UTCDateTime("2007-08-10")
    time_before = 10
    time_after = 60
    # approx. location of collapse
    latitude = 39.462
    longitude = -111.228
    max_dist = 150  # max distance from event (km) to keep station

    def download_events(self):
        """ Just copy the events into a directory. """
        cat = obspy.read_events(str(base_path / self.name / "events.xml"))
        catalog_to_directory(cat, self.event_path)

    def _download_crandall(self):
        """ download waveform/station info for dataset. """
        bank = WaveBank(self.waveform_path)
        domain = CircularDomain(
            self.latitude,
            self.longitude,
            minradius=0,
            maxradius=kilometers2degrees(self.max_dist),
        )
        cat = obspy.read_events(str(base_path / self.name / "events.xml"))
        df = events_to_df(cat)
        for _, row in df.iterrows():
            starttime = row.time - self.time_before
            endtime = row.time + self.time_after
            restrictions = Restrictions(
                starttime=UTC(starttime),
                endtime=UTC(endtime),
                minimum_length=0.90,
                minimum_interstation_distance_in_m=100,
                channel_priorities=["HH[ZNE]", "BH[ZNE]"],
                location_priorities=["", "00", "01", "--"],
            )
            kwargs = dict(
                domain=domain,
                restrictions=restrictions,
                mseed_storage=str(self.waveform_path),
                stationxml_storage=str(self.station_path),
            )
            MassDownloader(providers=[self._download_client]).download(**kwargs)
            # ensure data have actually been downloaded
            bank.update_index()
            assert not bank.read_index(starttime=starttime, endtime=endtime).empty

    download_stations = _download_crandall
    download_waveforms = _download_crandall


class Bingham(DataSet):
    """
    The Bingham Canyon dataset includes waveforms recorded during and after
    the Manefay Slide, one of the largest anthropogenic landslides ever
    recorded (https://bit.ly/2bsRsyR).Fortunately, due to close monitoring,
    no one was injured.

    I have personally modified and added some phase picks to the events.
    Without closer examination, this dataset should be used for testing
    and demonstration purposes only.
    """

    name = "bingham"
    time_before = 10
    time_after = 60
    # define spatial extents variables (center of pit)
    latitude = 40.53829
    longitude = -112.149_506
    max_dist = 20  # distance in km

    def download_events(self):
        """ Simply copy events from base directory. """
        cat = obspy.read_events(str(base_path / self.name / "events.xml"))
        catalog_to_directory(cat, self.event_path)

    def _download_bingham(self):
        """ Use obspy's mass downloader to get station/waveforms data. """
        bank = WaveBank(self.waveform_path)
        domain = CircularDomain(
            self.latitude,
            self.longitude,
            minradius=0,
            maxradius=kilometers2degrees(self.max_dist),
        )
        chan_priorities = ["HH[ZNE]", "BH[ZNE]", "EL[ZNE]", "EN[ZNE]"]
        cat = obspy.read_events(str(base_path / self.name / "events.xml"))
        df = events_to_df(cat)
        for _, row in df.iterrows():
            starttime = row.time - self.time_before
            endtime = row.time + self.time_after
            restrictions = Restrictions(
                starttime=UTC(starttime),
                endtime=UTC(endtime),
                minimum_length=0.90,
                minimum_interstation_distance_in_m=100,
                channel_priorities=chan_priorities,
                location_priorities=["", "00", "01", "--"],
            )
            kwargs = dict(
                domain=domain,
                restrictions=restrictions,
                mseed_storage=str(self.waveform_path),
                stationxml_storage=str(self.station_path),
            )
            MassDownloader(providers=[self._download_client]).download(**kwargs)
            # ensure data were downloaded
            bank.update_index()
            assert not bank.read_index(starttime=starttime, endtime=endtime).empty

        # update wavebank
        WaveBank(self.waveform_path).update_index()

    download_stations = _download_bingham
    download_waveforms = _download_bingham


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
