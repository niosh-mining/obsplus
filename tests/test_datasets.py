"""
Tests for the datasets
"""
import os
import random
import shutil
import string
import tempfile
from contextlib import suppress
from pathlib import Path

import obspy
import pytest

import obsplus
import obsplus.utils.dataset
from obsplus.constants import DATA_TYPES
from obsplus.datasets.dataset import DataSet
from obsplus.exceptions import (
    MissingDataFileError,
    FileHashChangedError,
    DataVersionError,
)
from obsplus.interfaces import WaveformClient, EventClient, StationClient


def make_dummy_dataset(cls_name="dummy", cls_version="0.1.0"):
    """ Create a dummy dataset and return cls definition. """

    class DummyDataset(DataSet):
        name = cls_name
        version = cls_version

        source_path = Path(tempfile.mkdtemp())

        def download_events(self) -> None:
            self.data_file = self.event_path / "dummy_file.txt"
            self.data_file1 = self.event_path / "dummy_file1.txt"
            os.makedirs(self.event_path, exist_ok=True)
            with open(self.data_file, "w") as f:
                f.write("test")
            with open(self.data_file1, "w") as f:
                f.write("abcd")

        def download_stations(self) -> None:
            os.makedirs(self.station_path, exist_ok=True)

        def download_waveforms(self) -> None:
            os.makedirs(self.waveform_path, exist_ok=True)

        def adjust_data(self) -> None:
            os.remove(self.data_file)
            with open(self.data_file1, "w") as f:
                f.write("efgh")

        def _log(self, msg):
            """Don't pring anything!"""

        @classmethod
        def cleanup(cls):
            """ Remove dataset source files, data files, and unregister."""
            self = cls()
            for path in [self.data_path, self.source_path]:
                with suppress(FileNotFoundError):
                    shutil.rmtree(str(path))
            DataSet._datasets.pop(self.__class__.name.lower(), None)

    return DummyDataset


@pytest.fixture(scope="session", params=list(DataSet._datasets))
def dataset(request):
    """ laod in the datasets """
    return DataSet._datasets[request.param]


@pytest.mark.dataset
class TestDatasets:
    """Generic tests for all loaded datasets."""

    client_types = {
        "event": EventClient,
        "station": StationClient,
        "waveform": WaveformClient,
    }

    @pytest.fixture(scope="class")
    def new_dataset(self, dataset, tmpdir_factory) -> DataSet:
        """
        Init a copy of each dataset and set its base directory to a temp
        directory to force download.
        """
        td = Path(tmpdir_factory.mktemp("datasets"))
        ds = dataset(base_path=td)
        return ds

    @pytest.fixture(scope="class")
    def datafetcher(self, new_dataset):
        """ call the dataset (forces download) and return fetcher. """
        return new_dataset.get_fetcher()

    def test_clients(self, datafetcher):
        """ Each dataset should have waveform, event, and station clients """
        for name, ctype in self.client_types.items():
            obj = getattr(datafetcher, name + "_client", None)
            assert isinstance(obj, ctype) or obj is None

    def test_directory_created(self, new_dataset):
        """ ensure the new directory was created. """
        for name in DATA_TYPES:
            path = getattr(new_dataset, name + "_path")
            assert path.exists()
            assert path.is_dir()

    def test_readme_created(self, new_dataset):
        """ ensure the readme was created. """
        path = new_dataset.data_path / "readme.txt"
        assert path.exists()

    def test_new_dataset(self, tmpdir):
        """ ensure a new dataset can be created and creates default paths """
        path = Path(tmpdir)
        inv = obspy.read_inventory()

        class NewDataSet(DataSet):
            name = "test_dataset"
            base_path = path
            version = "0.0.0"

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


class TestBasic:
    """ Basic misc. tests for dataset. """

    def test_all_files_copied(self, tmp_path):
        """
        When the download logic fires all files in the source
        should be copied.
        """
        ds = obsplus.copy_dataset("default_test", tmp_path)
        # iterate top level files and assert each was copied
        for tlf in ds.data_files:
            expected = ds.data_path / tlf.name
            assert expected.exists()

    def test_get_fetcher(self, ta_dataset):
        """ ensure a datafetcher can be created. """
        fetcher = ta_dataset.get_fetcher()
        assert isinstance(fetcher, obsplus.Fetcher)

    def test_can_copy_twice(self, ta_dataset, tmp_path):
        """ copying a dataset to the same location twice should work. """
        ta_dataset.copy_to(tmp_path)
        ta_dataset.copy_to(tmp_path)
        assert Path(tmp_path).exists()


class TestDatasetDownloadMemory:
    """
    Once downloaded datasets should remember to where they were
    downloaded. This allows datasets to live in multiple places.
    """

    expected_default_data_path = Path().home() / "opsdata"
    cls_name = "download_class_test"

    def read_saved_file_path(self, ds):
        """Read the contents of the saved datapath file."""
        path = ds._path_to_saved_path_file
        with path.open() as fi:
            contents = fi.read().rstrip()
        return contents

    @pytest.fixture
    def dummy_dataset_class(self):
        """Return a dummy dataset for testing, cleanup after it."""
        randstr = "".join(random.sample(string.ascii_uppercase, 12))
        name = "TESTDATASET" + randstr
        DS = make_dummy_dataset(name)
        yield DS
        DS.cleanup()

    def test_datasets_remember_download(self, dummy_dataset_class, tmp_path):
        """
        Datasets should remember where they have downloaded data.

        This enables users to store data in places other than the default, on
        a per-dataset basis.
        """
        # simply download a new dataset with a specified path. Init new
        newdata1 = dummy_dataset_class(base_path=tmp_path)
        # the data source should share the same data_path
        newdata2 = dummy_dataset_class()
        assert newdata1.data_path == newdata2.data_path

    def test_default_used_when_old_deleted(self, tmp_path, dummy_dataset_class):
        """ Ensure the default is used if the saved datapath is deleted. """
        newdata1 = dummy_dataset_class(base_path=tmp_path)
        newdata2 = dummy_dataset_class()
        # these should be the same data path
        assert newdata2.data_path == newdata1.data_path
        # until the data path gets deleted
        newdata1.delete_data_directory()
        # then it should default back to normal directory
        dataset3 = dummy_dataset_class()
        assert dataset3.data_path != newdata1.data_path
        assert self.expected_default_data_path == dataset3.data_path.parent
        dataset3.delete_data_directory()

    def test_copying_doesnt_change_file(self, tmp_path, dummy_dataset_class):
        """Making a copy of the dataset shouldn't change the stored path."""
        ds = dummy_dataset_class(base_path=tmp_path)
        path1 = self.read_saved_file_path(ds)
        ds2 = ds.copy()
        path2 = self.read_saved_file_path(ds2)
        ds3 = ds.copy_to(tmp_path)
        path3 = self.read_saved_file_path(ds3)
        assert path1 == path2 == path3


class TestCopyDataset:
    """ tests for copying datasets. """

    def test_no_new_data_downloaded(self, monkeypatch):
        """
        No data should be downloaded when copying a dataset that
        has already been downloaded.
        """
        # this ensures dataset is loaded
        ds = obsplus.load_dataset("bingham_test")
        cls = ds.__class__

        def fail(*args, **kwargs):
            pytest.fail()  # this should never be called

        # monkey patch download methods with fail to ensure they aren't called
        for name in DATA_TYPES:
            attr = f"download_{name}s"
            monkeypatch.setattr(cls, attr, fail)

        # if this proceeds without downloading data the test passes
        new_ds = obsplus.utils.dataset.copy_dataset("bingham_test")
        assert isinstance(new_ds, DataSet)

    def test_copy_dataset_with_dataset(self):
        """ ensure a dataset can be the first argument to copy_dataset """
        ds = obsplus.load_dataset("bingham_test")
        out = obsplus.utils.dataset.copy_dataset(ds)
        assert isinstance(out, DataSet)
        assert out.name == ds.name
        assert out.data_path != ds.data_path

    def test_copy_unknown_dataset(self):
        """ ensure copying a dataset that doesn't exit raises. """
        with pytest.raises(ValueError):
            obsplus.load_dataset("probably_not_a_real_dataset")

    def test_str_and_repr(self):
        """ ensure str is returned from str and repr """
        ds = obsplus.load_dataset("bingham_test")
        assert isinstance(str(ds), str)  # these are dumb COV tests
        assert isinstance(ds.__repr__(), str)


class TestFileHashing:
    """Ensure a hash can be created for each file in a directory."""

    @pytest.fixture
    def copied_crandall(self, tmpdir_factory):
        """
        Copy the crandall_test ds to a new directory, create fresh hash
        and return.
        """
        newdir = Path(tmpdir_factory.mktemp("new_ds"))
        ds = obsplus.load_dataset("crandall_test").copy_to(newdir)
        ds.create_sha256_hash()
        return ds

    @pytest.fixture
    def crandall_deleted_file(self, copied_crandall):
        """ Delete a file """
        path = copied_crandall.data_path
        for mseed in path.rglob("*.mseed"):
            os.remove(mseed)
            break
        return copied_crandall

    @pytest.fixture
    def crandall_changed_file(self, copied_crandall):
        """ Change a file (after hash has already run) """
        path = copied_crandall.data_path
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
        copied_crandall.check_hashes()

    def test_missing_file_found(self, crandall_deleted_file):
        """ Ensure a missing file is found. """
        with pytest.raises(MissingDataFileError):
            crandall_deleted_file.check_hashes()

    def test_bad_hash(self, crandall_changed_file):
        """ Test that when a file was changed the hash function raises. """
        # should not raise if the file has changed
        crandall_changed_file.check_hashes()
        # raise an error if checking for it
        with pytest.raises(FileHashChangedError):
            crandall_changed_file.check_hashes(check_hash=True)


class TestVersioning:
    """ Verify logic for checking dataset versions works """

    # Helper Functions
    def check_dataset(self, ds, redownloaded=True):
        """Perform sanity-checks on a dataset."""
        if redownloaded:
            assert ds.data_file.exists()
            with open(ds.data_file1) as f:
                assert f.read() == "abcd"
        else:
            assert not ds.data_file.exists()
            with open(ds.data_file1) as f:
                assert f.read() == "efgh"

    # Fixtures
    @pytest.fixture(scope="class")
    def dataset(self):
        """Create a stupidly simple dataset."""
        ds_cls = make_dummy_dataset(cls_name="dummy", cls_version="1.0.0")
        yield ds_cls
        ds_cls.cleanup()

    @pytest.fixture
    def dummy_dataset(self, tmpdir_factory, dataset):
        """
        Instantiate our simple dataset for the first data download.
        """
        newdir = Path(tmpdir_factory.mktemp("new_ds"))
        ds = dataset(base_path=newdir)
        ds.create_sha256_hash()
        return ds

    @pytest.fixture  # These should be able to be combined somehow, I would think...
    def proper_version(self, dummy_dataset):
        """
        Make sure there is a version file with the correct version number from
        what is attached to the DataSet.
        """
        version = "1.0.0"
        with dummy_dataset._version_path.open("w") as fi:
            fi.write(version)
        dummy_dataset.adjust_data()
        return dummy_dataset

    @pytest.fixture
    def low_version(self, dummy_dataset):
        """
        Make sure the version file has a lower version number than what is
        attached to the DataSet.
        """
        version = "0.0.0"
        with open(dummy_dataset._version_path, "w") as fi:
            fi.write(version)
        dummy_dataset.adjust_data()
        return dummy_dataset

    @pytest.fixture
    def high_version(self, dummy_dataset):
        """Return a dummy dataset with a large version."""
        version = "2.0.0"
        with open(dummy_dataset._version_path, "w") as fi:
            fi.write(version)
        dummy_dataset.adjust_data()
        return dummy_dataset

    @pytest.fixture
    def no_version(self, dummy_dataset):
        """Return a dummy dataset with no version info."""
        path = dummy_dataset._version_path
        if path.exists():
            os.remove(path)
        dummy_dataset.adjust_data()
        return dummy_dataset

    @pytest.fixture
    def re_download(self, dummy_dataset):
        """Re-download the dummy dataset and return."""
        path = dummy_dataset._version_path
        if path.exists():
            os.remove(path)
        for path in [
            dummy_dataset.event_path,
            dummy_dataset.station_path,
            dummy_dataset.waveform_path,
        ]:
            shutil.rmtree(path)
        return dummy_dataset

    @pytest.fixture
    def corrupt_version(self, dummy_dataset):
        """Return a dummy dataset with a corrupt version."""
        with open(dummy_dataset._version_path, "w") as fi:
            fi.write("abcd")
        dummy_dataset.adjust_data()
        return dummy_dataset

    # Tests <- It's probably possible to combine some of these, somehow...
    def test_version_matches(self, proper_version, dataset):
        """
        Try re-loading the dataset and verify that it is possible with a
        matching version number.
        """
        dataset(base_path=proper_version.data_path.parent)
        # Make sure the data were not re-downloaded
        self.check_dataset(proper_version, redownloaded=False)

    def test_version_greater(self, high_version, dataset):
        """ Make sure a warning is issued if the version number is too high """
        with pytest.warns(UserWarning):
            dataset(base_path=high_version.data_path.parent)
        # Make sure the data were not re-downloaded
        self.check_dataset(high_version, redownloaded=False)

    def test_version_less(self, low_version, dataset):
        """ Make sure an exception gets raised if the version number is too low """
        with pytest.raises(DataVersionError):
            dataset(base_path=low_version.data_path.parent)
        # Make sure the data were not re-downloaded
        self.check_dataset(low_version, redownloaded=False)

    def test_no_version(self, no_version, dataset):
        """ Make sure a missing version file will trigger a re-download """
        with pytest.warns(UserWarning):
            dataset(base_path=no_version.data_path.parent)
        # Make sure the data were re-downloaded
        self.check_dataset(no_version, redownloaded=True)

    def test_deleted_files(self, re_download, dataset):
        """ Make sure the dataset can be re-downloaded if proper files were deleted """
        dataset(base_path=re_download.data_path.parent)
        # Make sure the data were re-downloaded
        self.check_dataset(re_download, redownloaded=True)

    def test_corrupt_version_file(self, corrupt_version, dataset):
        """ Make sure a bogus version file will trigger a re-download """
        with pytest.warns(UserWarning):
            dataset(base_path=corrupt_version.data_path.parent)
        # Make sure the data were re-downloaded
        self.check_dataset(corrupt_version, redownloaded=True)

    def test_listed_files(self, low_version, dataset):
        """ Make sure the download directory is listed in exception. """
        expected = str(low_version.data_path)
        with pytest.raises(DataVersionError) as e:
            dataset(base_path=low_version.data_path.parent)
        assert expected in str(e.value.args[0])

    def test_doesnt_delete_extra_files(self, no_version, dataset):
        """
        Make sure an extra file that was added doesn't get harmed by the
        re-download process.
        """
        path = no_version.station_path / "test.txt"
        with open(path, "w") as f:
            f.write("ijkl")
        with pytest.warns(UserWarning):  # this emits a warning
            dataset(base_path=no_version.data_path.parent)
        # First make sure the data were re-downloaded
        self.check_dataset(no_version, redownloaded=True)
        # Now make sure the dummy file didn't get destroyed
        with open(path) as f:
            assert f.read() == "ijkl"


#
# class TestDatasetUtils:
#     """ Tests for dataset utilities. """
#
#     def test_base_directory_created(self, tmp_path: Path):
#         """ Ensure the base directory is created. """
#         expected_path = tmp_path / "opsdata"
#         assert not expected_path.exists()
#         out = DataSet._get_opsdata_path(expected_path)
#         assert out.exists()
#         assert (out / "README.txt").exists()
