"""
Tests for the datasets
"""
from collections import defaultdict
import os
from pathlib import Path

import obspy
import pytest

import obsplus
from obsplus.datasets.dataloader import DataSet
from obsplus.interfaces import WaveformClient, EventClient, StationClient
from obsplus.exceptions import MissingDataFileError, FileHashChangedError


@pytest.fixture(scope="session", params=list(DataSet.datasets))
def dataset(request):
    """ laod in the datasets """
    return DataSet.datasets[request.param]


names = ("event", "station", "waveform")


@pytest.mark.dataset
class TestDatasets:
    """ generic tests for all loaded datasets. These require downloaded a few
     mb of data. """

    client_types = (EventClient, StationClient, WaveformClient)

    @pytest.fixture(scope="class")
    def new_dataset(self, dataset, tmpdir_factory) -> DataSet:
        """ init a copy of each dataset and set its base directory to a temp
        directory to force download. """
        td = Path(tmpdir_factory.mktemp("datasets"))
        ds = dataset(base_path=td)
        return ds

    @pytest.fixture(scope="class")
    def datafetcher(self, new_dataset):
        """ call the dataset (forces download) and return fetcher. """
        return new_dataset.get_fetcher()

    @pytest.fixture(scope="class")
    def downloaded_dataset(self, new_dataset, datafetcher):
        """
        Data should be downloaded and loaded into memory.
        Monkey patch all methods that load data with pytest fails.
        """
        ds = new_dataset
        old_load_funcs = ds._load_funcs
        old_download_funcs = {}

        def _fail(*args, **kwargs):
            pytest.fail("this should not get called!")

        for name in names:
            func_name = "download_" + name + "s"
            old_download_funcs[name] = getattr(ds, func_name)
            setattr(ds, func_name, _fail)
        ds._load_funcs = defaultdict(lambda: _fail)
        yield ds
        # reset monkey patching
        ds._load_funcs = old_load_funcs
        for name in names:
            func_name = "download_" + name + "s"
            setattr(ds, func_name, old_download_funcs[name])

    def test_clients(self, datafetcher):
        """ Each dataset should have waveform, event, and station clients """
        for name, ctype in zip(names, self.client_types):
            obj = getattr(datafetcher, name + "_client", None)
            assert isinstance(obj, ctype) or obj is None

    def test_directory_created(self, new_dataset):
        """ ensure the new directory was created. """
        for name in names:
            path = getattr(new_dataset, name + "_path")
            assert path.exists()
            assert path.is_dir()

    def test_readme_created(self, new_dataset):
        """ ensure the readme was created. """
        path = new_dataset.path / "readme.txt"
        assert path.exists()

    def test_new_dataset(self, tmpdir):
        """ ensure a new dataset can be created and creates default paths """
        path = Path(tmpdir)
        inv = obspy.read_inventory()

        class NewDataSet(DataSet):
            name = "test_dataset"
            base_path = path

            def download_stations(self):
                self.station_path.mkdir(exist_ok=True, parents=True)
                path = self.station_path / "inv.xml"
                inv.write(str(path), "stationxml")

        # now the new dataset should be loadable
        ds = obsplus.load_dataset(NewDataSet.name)
        # and the empty directories for event, waveforms, station should exist
        assert not ds.events_need_downloading
        assert not ds.stations_need_downloading
        assert not ds.waveforms_need_downloading


class TestCopyDataset:
    """ tests for copying datasets. """

    def test_no_new_data_downloaded(self, monkeypatch):
        """ no data should be downloaded when copying a dataset that
        has already been downloaded. """
        # this ensures dataset is loaded
        ds = obsplus.load_dataset("bingham")
        cls = ds.__class__

        def fail(*args, **kwargs):
            pytest.fail()  # this should never be called

        # monkey patch download methods with fail to ensure they arent called
        for name in names:
            attr = f"download_{name}s"
            monkeypatch.setattr(cls, attr, fail)

        # if this proceeds without downloading data the test passes
        new_ds = obsplus.copy_dataset("bingham")
        assert isinstance(new_ds, DataSet)

    def test_copy_dataset_with_dataset(self):
        """ ensure a dataset can be the first argument to copy_dataset """
        ds = obsplus.load_dataset("bingham")
        out = obsplus.copy_dataset(ds)
        assert isinstance(out, DataSet)
        assert out.name == ds.name
        assert out.path != ds.path

    def test_copy_unknown_dataset(self):
        """ ensure copying a dataset that doesn't exit raises. """
        with pytest.raises(ValueError):
            obsplus.load_dataset("probably_not_a_real_dataset")

    def test_str_and_repr(self):
        """ ensure str is returned from str and repr """
        ds = obsplus.load_dataset("bingham")
        assert isinstance(str(ds), str)  # these are dumb COV tests
        assert isinstance(ds.__repr__(), str)


class TestMD5Hash:
    """ Ensure a MD5 hash can be created of directory contents. """

    @pytest.fixture
    def copied_crandall(self, tmpdir_factory):
        """ Copy the crandall ds to a new directory, create fresh hash
        and return. """
        newdir = Path(tmpdir_factory.mktemp("new_ds"))
        ds = obsplus.load_dataset("crandall").copy_to(newdir)
        ds.create_md5_hash()
        return ds

    @pytest.fixture
    def crandall_deleted_file(self, copied_crandall):
        """ Delete a file """
        path = copied_crandall.path
        for mseed in path.rglob("*.mseed"):
            os.remove(mseed)
            break
        return copied_crandall

    @pytest.fixture
    def crandall_changed_file(self, copied_crandall):
        """ Change a file (after hash has already run) """
        path = copied_crandall.path
        for mseed in path.rglob("*.mseed"):
            st = obspy.read(str(mseed))
            for tr in st:
                tr.data = tr.data * 2
            st.write(str(mseed), "mseed")
            break
        return copied_crandall

    def test_good_hash(self, copied_crandall):
        """ Test hashing the file contents. """
        # when nothing has changed check hash should work silently
        copied_crandall.check_files()

    def test_missing_file_found(self, crandall_deleted_file):
        """ Ensure a missing file is found. """
        with pytest.raises(MissingDataFileError):
            crandall_deleted_file.check_files()

    def test_bad_hash(self, crandall_changed_file):
        """ Test that when a file was changed the hash function raises. """
        # should not raise if the file has changed
        crandall_changed_file.check_files()
        # raise an error if checking for it
        with pytest.raises(FileHashChangedError):
            crandall_changed_file.check_files(check_hash=True)
