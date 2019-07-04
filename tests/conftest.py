"""
pytest configuration for obsplus
"""
import copy
import glob
import os
import shutil
import tempfile
import typing
import warnings
from os.path import basename
from os.path import join, dirname, abspath, exists
from pathlib import Path

import numpy as np
import obspy
import pytest
from obspy.core.event.base import ResourceIdentifier

import obsplus
import obsplus.datasets.utils
import obsplus.events.utils

# ------------------------- define constants

# path to the test directory

TEST_PATH = abspath(dirname(__file__))
# path to the package directory
PKG_PATH = dirname(TEST_PATH)
# path to the test data directory
TEST_DATA_PATH = join(TEST_PATH, "test_data")
# the path to the setup scripts
SETUP_PATH = join(TEST_PATH, "setup_code")
# Directory where catalogs are stored
CATALOG_DIRECTORY = join(TEST_DATA_PATH, "test_catalogs")
# Directory containing test inventories
INVENTORY_DIRECTORY = join(TEST_DATA_PATH, "test_inventories")
# Directory containing test data for grids
GRID_DIRECTORY = join(TEST_DATA_PATH, "test_grid_inputs")
# get a list of cat_name file paths
catalogs = glob.glob(join(CATALOG_DIRECTORY, "*xml"))
# get a list of stations file paths
inventories = glob.glob(join(INVENTORY_DIRECTORY, "*"))
# init a dict for storing event id and events objects in
eve_id_cache = {}

# path to obsplus datasets
DATASETS = join(dirname(obsplus.__file__), "datasets")


# Monkey patch the resource_id to avoid emmitting millions of warnings
# TODO Remove this when obspy 1.2 is released
old_func = ResourceIdentifier._get_similar_referred_object


def _func(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_func(*args, **kwargs)


ResourceIdentifier._get_similar_referred_object = _func


# ------------------------------ helper functions


class ObspyCache:
    """
    A dict-like class for lazy loading from the test directory
    """

    def __init__(self, test_dir: str, load_func):
        self.load_func = load_func
        self._path = join(TEST_DATA_PATH, test_dir)
        self._file_paths = set(glob.glob(join(self._path, "*")))
        # add the file names and stripped names to keys
        self.keys = {basename(x).split(".")[0]: x for x in self._file_paths}
        self.keys.update({basename(x): x for x in self._file_paths})
        self.file_paths = list(self.keys.keys())
        self._objects = {}

    def __getattr__(self, item):
        if item in self.keys:
            return self._load_object(item)
        else:
            raise AttributeError

    def __getitem__(self, item):
        if item in self.keys:
            return self._load_object(item)
        else:
            raise IndexError

    def _load_object(self, item):
        if item not in self._objects:
            self._objects[item] = self.load_func(self.keys[item])
        try:
            return self._objects[item].copy()
        except AttributeError:
            return copy.deepcopy(self._objects[item])


def collect_catalogs():
    """
    a dictionary of catalogs by file name, useful for tests that
    target a specific cat_name
    """

    def _get_origin_time(cat):
        ori = obsplus.get_preferred(cat, "origin")
        return ori.time

    out = {}
    for cat_path in catalogs:
        cat = obspy.read_events(cat_path)
        # sort events by origin time
        cat.events.sort(key=_get_origin_time)
        out[os.path.basename(cat_path).split(".")[0]] = cat
        # add to cache and assert cat_name resource id is unique
        cat_id = cat.resource_id.id
        assert cat_id not in eve_id_cache  # test catalogs must have unique id
        eve_id_cache[cat_id] = len(cat)

    # get catalog from datasets
    out["kemmerer"] = obsplus.load_dataset("kemmerer").event_client.get_events()
    out["bingham"] = obsplus.load_dataset("bingham").event_client.get_events()
    out["crandall"] = obsplus.load_dataset("crandall").event_client.get_events()
    return out


cat_dict = collect_catalogs()
waveform_cache_obj = ObspyCache("waveforms", obspy.read)
event_cache_obj = ObspyCache("qml_files", obspy.read_events)
test_catalog_cache_obj = ObspyCache(CATALOG_DIRECTORY, obspy.read_events)
station_cache_obj = ObspyCache(INVENTORY_DIRECTORY, obspy.read_inventory)


# -------------------- collection of test cases


class StreamTester:
    """ A collection of methods for testing waveforms. """

    @staticmethod
    def streams_almost_equal(st1, st2):
        """
        Return True if two streams are almost equal.
        Will only look at default params for stats objects.
        Parameters
        ----------
        st1
            The first stream.
        st2
            The second stream.
        """

        stats_attrs = (
            "starttime",
            "endtime",
            "sampling_rate",
            "network",
            "station",
            "location",
            "channel",
        )

        st1, st2 = st1.copy(), st2.copy()
        st1.sort()
        st2.sort()
        for tr1, tr2 in zip(st1, st2):
            stats1 = {x: tr1.stats[x] for x in stats_attrs}
            stats2 = {x: tr2.stats[x] for x in stats_attrs}
            if not stats1 == stats2:
                return False
            if not np.all(np.isclose(tr1.data, tr2.data)):
                return False
        return True


class DataSet(typing.NamedTuple):
    """ A data class for storing info about test cases """

    base = None
    quakeml = None
    events_csv = None
    inventory = None
    station_csv = None
    waveform_path = None
    inventory_path = None


@pytest.fixture(scope="session")
def ta_dataset():
    """ Load the small TA test case into a dataset """
    return obsplus.load_dataset("TA")


@pytest.fixture(scope="session")
def kemmerer_dataset():
    """ Load the kemmerer test case """
    return obsplus.load_dataset("kemmerer")


@pytest.fixture(scope="session")
def bingham_dataset():
    """ load the bingham dataset """
    return obsplus.load_dataset("bingham")


@pytest.fixture(scope="session")
def bingham_inventory(bingham_dataset):
    """ load the bingham tests case """
    return bingham_dataset.station_client


@pytest.fixture(scope="session")
def bingham_stream_dict(bingham_dataset):
    """ return a dict where keys are event_id, vals are streams """
    fetcher = bingham_dataset.get_fetcher()
    return dict(fetcher.yield_event_waveforms(10, 50))


@pytest.fixture(scope="session")
def crandall_dataset():
    """ load the crandall canyon dataset. """
    return obsplus.load_dataset("crandall")


@pytest.fixture(scope="session")
def crandall_fetcher(crandall_dataset):
    """ Return a Fetcher from the crandall dataset. """
    return crandall_dataset.get_fetcher()


@pytest.fixture(scope="session")
def crandall_data_array(crandall_fetcher):
    """ return a data array (with attached picks) of crandall dataset. """
    cat = crandall_fetcher.event_client.get_events()
    st_dict = crandall_fetcher.get_event_waveforms(10, 50)
    dar = obsplus.obspy_to_array_dict(st_dict)[40]
    dar.ops.attach_events(cat)
    return dar


@pytest.fixture(scope="session")
def crandall_bank(crandall_dataset):
    return obsplus.WaveBank(crandall_dataset.waveform_client)


# ------------------------- session fixtures


@pytest.fixture(scope="session")
def stream_tester():
    """ return the StreamTester. """
    return StreamTester


@pytest.yield_fixture(scope="module")
def temp_dir():
    """ create a temporary archive directory """
    td = tempfile.mkdtemp()
    yield td
    if exists(td):
        shutil.rmtree(td)


@pytest.fixture(scope="session", params=cat_dict.values())
def test_catalog(request):
    """
    return a list of test events (as catalogs) from
    quakeml saved on disk
    """
    cat = request.param
    return cat


@pytest.fixture(scope="session")
def catalog_dic():
    """ a dictionary of catalogs based on name of cat_name file """
    return cat_dict


@pytest.fixture(scope="session", params=inventories)
def test_inventory(request):
    """ return a list of test inventories from
    stationxml files saved on disk """
    inv = obspy.read_inventory(request.param)
    return inv


@pytest.fixture(scope="session")
def cd_2_test_directory():
    """ cd to test directory, then cd back """
    back = os.getcwd()
    there = dirname(__file__)
    os.chdir(there)
    yield  # there and back again
    os.chdir(back)


@pytest.fixture(scope="session")
def ta_archive(ta_dataset):
    """ make sure the TA archive, generated with the setup_test_archive
    script, has been downloaded, else download it """
    return Path(obsplus.WaveBank(ta_dataset.waveform_client).index_path).parent


@pytest.fixture(scope="session")
def kem_archive(kemmerer_dataset):
    """ download the kemmerer data (will take a few minutes but only
     done once) """
    return Path(obsplus.WaveBank(kemmerer_dataset.waveform_client).index_path).parent


@pytest.fixture(scope="session")
def kem_fetcher():
    """ return a wavefetcher of the kemmerer dataset, download if needed """
    return obsplus.load_dataset("kemmerer").get_fetcher()


@pytest.fixture(scope="class")
def class_tmp_dir(tmpdir_factory):
    """ create a temporary directory on the class scope """
    return tmpdir_factory.mktemp("base")


@pytest.fixture(scope="class")
def tmp_ta_dir(class_tmp_dir):
    """ Make a temp copy of the TA bank """
    bank = obsplus.load_dataset("TA").waveform_client
    path = dirname(dirname(bank.index_path))
    out = os.path.join(class_tmp_dir, "temp")
    shutil.copytree(path, out)
    yield out


@pytest.fixture(scope="session")
def bingham_bank_path(tmpdir_factory):
    """ Create  bank structure using Bingham dataset """
    tmpdir = tmpdir_factory.mktemp("data")
    obsplus.datasets.utils.copy_dataset("bingham", tmpdir)
    return str(tmpdir)


@pytest.fixture(scope="class", params=waveform_cache_obj.keys)
def waveform_cache_stream(request):
    """
    Return the each stream in the cache
    """
    return waveform_cache_obj[request.param]


@pytest.fixture(scope="class")
def waveform_cache_trace(waveform_cache_stream):
    """
    Return the first trace of each test stream
    """
    return waveform_cache_stream[0]


@pytest.fixture(scope="session")
def waveform_cache():
    """
    Return the waveform cache object for tests to select which waveform
    should be used by name.
    """
    return waveform_cache_obj


# TODO these next two fixtures are similar, see if they can be merged


@pytest.fixture(scope="session")
def event_cache():
    """
    Return the event cache (qml) object for tests to select particular
    quakemls.
    """
    return event_cache_obj


@pytest.fixture(scope="session")
def catalog_cache():
    """ Return a cache from the test_catalogs. """
    return test_catalog_cache_obj


@pytest.fixture(params=glob.glob(join(TEST_DATA_PATH, "qml2merge", "*")))
def qml_to_merge_paths(request):
    """
    Returns a path to each qml2merge directory, which contains two catalogs
    for merging together.
    """
    return request.param


@pytest.fixture(scope="class")
def qml_to_merge_basic():
    """ Return a path to the basic merge qml dataset. """
    out = glob.glob(join(TEST_DATA_PATH, "qml2merge", "*2017-01-06T16-15-14"))
    return out[0]


@pytest.fixture(scope="session")
def station_cache():
    """ Return the station cache. """
    return station_cache_obj


@pytest.fixture(scope="session", params=station_cache_obj.keys)
def station_cache_inventory(request):
    """ Return the test inventories. """
    return station_cache_obj[request.param]


@pytest.fixture(scope="session")
def grid_path():
    """ Return the path to the grid inputs """
    return GRID_DIRECTORY


# -------------- configure test runner


def pytest_addoption(parser):
    parser.addoption(
        "--datasets",
        action="store_true",
        dest="datasets",
        default=False,
        help="enable dataset tests",
    )


def pytest_configure(config):
    if not config.option.datasets:
        setattr(config.option, "markexpr", "not dataset")
