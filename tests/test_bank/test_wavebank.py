""" test for core functionality of wavebank """
import copy
import functools
import glob
import os
import pathlib
import pickle
import shutil
import tempfile
import time
import types
from concurrent.futures import as_completed, ProcessPoolExecutor
from contextlib import suppress

from os.path import join
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose as np_assert
import obspy
import obspy.clients.fdsn
import pandas as pd
import pytest
from obspy import UTCDateTime as UTC

import obsplus
import obsplus.utils.bank
import obsplus.utils.dataset
import obsplus.utils.misc
import obsplus.utils.pd
from obsplus.bank.wavebank import WaveBank
from obsplus.constants import NSLC, EMPTYTD64, WAVEFORM_DTYPES
from obsplus.exceptions import BankDoesNotExistError, UnsupportedKeyword
from obsplus.utils.time import to_datetime64, to_timedelta64, to_utc
from obsplus.utils.testing import check_index_paths
from obsplus import get_reference_time

# ----------------------------------- Helper functions
from obsplus.utils.testing import ArchiveDirectory


def count_calls(instance, bound_method, counter_attr):
    """
    Given an instance, count how many times bound_method is called and
    store the result on instance in an attr named counter_attr.
    """
    # set counter
    setattr(instance, counter_attr, 0)

    @functools.wraps(bound_method)
    def wrapper(*args, **kwargs):
        out = bound_method(*args, **kwargs)
        # increment counter
        setattr(instance, counter_attr, getattr(instance, counter_attr) + 1)
        return out

    return wrapper


def strip_processing(st: obspy.Stream) -> obspy.Stream:
    """strip out processing from stats"""
    # TODO replace this when ObsPy #2286 gets merged
    for tr in st:
        tr.stats.pop("processing")
    return st


# ------------------------------ Fixtures


@pytest.fixture(scope="class")
def ta_bank(tmp_ta_dir):
    """init a bank on the test ta_test dataset"""
    bank_path = os.path.join(tmp_ta_dir, "waveforms")
    return WaveBank(bank_path)


@pytest.fixture(scope="function")
def ta_bank_index(ta_bank):
    """return the ta bank, but first update index"""
    ta_bank.update_index()
    return ta_bank


@pytest.fixture(scope="function")
def ta_bank_no_index(ta_bank):
    """return the ta bank, but first delete the index"""
    path = Path(ta_bank.index_path)
    if path.exists():
        os.remove(path)
    return ta_bank


@pytest.fixture(scope="function")
def ta_index(ta_bank_index):
    """return the ta index"""
    return ta_bank_index.read_index()


@pytest.fixture
def empty_bank(tmp_path):
    """init a bank with an empty directory, return"""
    return WaveBank(tmp_path)


@pytest.fixture
def duplicate_bank(tmp_path):
    """init a bank with duplicate waveforms."""
    # create two overlapping traces for each stream
    st1 = obspy.read()
    st1.trim(endtime=st1[0].stats.endtime - 10)
    st2 = obspy.read()
    for tr in st2:
        tr.stats.starttime = st1[0].stats.endtime - 10
    # get bank and save streams
    wbank = WaveBank(tmp_path)
    st1.write(str(tmp_path / "first.mseed"), "mseed")
    st2.write(str(tmp_path / "second.mseed"), "mseed")
    df = wbank.update_index().read_index()
    assert len(df) == 6, "there should be 6 files"
    return wbank


# ------------------------------ Tests


class TestBankBasics:
    """basic tests for Bank class"""

    low_version_str = "0.0.-1"

    # fixtures
    @pytest.fixture
    def default_bank_low_version(self, default_wbank, monkeypatch):
        """return the default bank with a negative version number."""
        # monkey patch obsplus version
        monkeypatch.setattr(obsplus, "__last_version__", self.low_version_str)
        # write index with negative version
        os.remove(default_wbank.index_path)
        default_wbank.update_index()
        assert default_wbank._index_version == self.low_version_str
        # restore correct version
        monkeypatch.undo()
        assert obsplus.__last_version__ != self.low_version_str
        assert Path(default_wbank.bank_path).exists()
        return default_wbank

    @pytest.fixture
    def cust_wbank_index_path(self, tmpdir_factory):
        """Path for a custom index location"""
        return tmpdir_factory.mktemp("custom_index") / ".index.h5"

    @pytest.fixture
    def cust_index_wbank(self, tmp_ta_dir, cust_wbank_index_path):
        """WaveBank that uses a custom index path"""
        bank_path = os.path.join(tmp_ta_dir, "waveforms")
        bank = WaveBank(bank_path, index_path=cust_wbank_index_path)
        bank.update_index()
        return bank

    @pytest.fixture
    def legacy_path_index(self, default_wbank, monkeypatch):
        """
        Overwrite 'read_index' to return an index with leading '/'s in the
        file paths.
        """
        ind = default_wbank.read_index()
        ind["path"] = "/" + ind["path"]

        def read_index(*args, **kwargs):
            return ind

        monkeypatch.setattr(default_wbank, "read_index", read_index)
        yield
        monkeypatch.undo()

    # tests
    def test_type(self, ta_bank):
        """make sure ta_test bank is a bank"""
        assert isinstance(ta_bank, WaveBank)

    def test_index(self, ta_bank_index):
        """ensure the index exists"""
        assert os.path.exists(ta_bank_index.index_path)
        assert isinstance(ta_bank_index.last_updated_timestamp, float)

    def test_custom_index_path(self, cust_index_wbank, cust_wbank_index_path):
        """ensure a custom index path can be used"""
        index_path = cust_index_wbank.index_path
        # Make sure the new path got passed correctly
        assert index_path == cust_wbank_index_path
        assert os.path.exists(index_path)
        assert isinstance(cust_index_wbank.last_updated_timestamp, float)
        # Make sure paths got written to the index properly
        check_index_paths(cust_index_wbank)

    def test_create_index(self, ta_bank_no_index):
        """make sure a fresh index can be created"""
        # test that just trying to get an index that doesnt exists creates it
        ta_bank_no_index.read_index()
        index_path = ta_bank_no_index.index_path
        assert os.path.exists(index_path)
        # make sure all file paths are in the index
        check_index_paths(ta_bank_no_index)

    def test_update_index_bumps_only_for_new_files(self, ta_bank_index):
        """
        Test that updating the index does not modify the last_updated
        time if no new files were added.
        """
        last_updated1 = ta_bank_index.last_updated_timestamp
        ta_bank_index.update_index()
        last_updated2 = ta_bank_index.last_updated_timestamp
        # updating should not get stamped unless files were added
        assert last_updated1 == last_updated2

    def test_pathlib_object(self, tmp_ta_dir):
        """ensure a pathlib object can be passed as first arg"""
        bank = WaveBank(pathlib.Path(tmp_ta_dir) / "waveforms")
        ind = bank.read_index()
        min_start = ind.starttime.min()
        td = np.timedelta64(600, "s")
        st = bank.get_waveforms(starttime=min_start, endtime=min_start + td)
        assert isinstance(st, obspy.Stream)

    def test_starttime_larger_than_endtime_raises(self, ta_bank):
        """
        If any function that reads index is given a starttime larger
        than the endtime a ValueError should be raised
        """
        with pytest.raises(ValueError) as e:
            ta_bank.read_index(starttime=10, endtime=1)
        e_msg = str(e.value.args[0])
        assert "starttime cannot be greater than endtime" in e_msg

    def test_correct_endtime_in_index(self, default_wbank):
        """ensure the index has times consistent with traces in waveforms"""
        index = default_wbank.read_index()
        st = obspy.read()
        starttimes = [to_datetime64(tr.stats.starttime) for tr in st]
        endtimes = [to_datetime64(tr.stats.endtime) for tr in st]
        assert min(starttimes) == index.starttime.min().to_datetime64()
        assert max(endtimes) == index.endtime.max().to_datetime64()

    def test_bank_can_init_bank(self, default_wbank):
        """WaveBank should be able to take a Wavebank as an input arg."""
        bank = obsplus.WaveBank(default_wbank)
        assert isinstance(bank, obsplus.WaveBank)

    def test_stream_is_not_instance(self):
        """A waveforms object should not be an instance of WaveBank."""
        assert not isinstance(obspy.read(), WaveBank)

    def test_min_version_recreates_index(self, default_bank_low_version):
        """
        If the min version is not met the index should be deleted and re-created.
        A warning should be issued.
        """
        bank = default_bank_low_version
        with pytest.warns(UserWarning):
            bank.update_index()
        assert bank._index_version == obsplus.__last_version__
        assert Path(bank.index_path).exists()

    def test_min_version_new_bank_recreates_index(self, default_bank_low_version):
        """
        A new bank should delete the old index and getting data from the bank
        should recreate it.
        """
        bank = default_bank_low_version
        assert bank._index_version == self.low_version_str
        # initing a new bank should warn and delete the old index
        with pytest.warns(UserWarning):
            bank2 = WaveBank(bank.bank_path)
        assert not Path(bank2.index_path).exists()
        bank2.get_waveforms()
        assert bank2._index_version != self.low_version_str
        assert bank2._index_version == obsplus.__last_version__
        assert Path(bank2.index_path).exists()

    def test_empty_bank_raises(self, tmpdir):
        """
        Test that an empty bank can be inited, but that an error is
        raised when trying to read its index.
        """
        path = Path(tmpdir) / "new"
        bank = WaveBank(path)
        # test that touching the index/meta data raises
        with pytest.raises(BankDoesNotExistError):
            bank.read_index()
        bank.put_waveforms(obspy.read(), update_index=True)
        assert len(bank.read_index()) == 3

    def test_update_index_returns_self(self, default_wbank):
        """ensure update index returns the instance for chaining."""
        out = default_wbank.update_index()
        assert out is default_wbank

    def test_index_time_types(self, default_wbank):
        """Ensure the time columns are np datetimes"""
        df = default_wbank.read_index()
        np_datetime_cols = df.select_dtypes(np.datetime64).columns
        assert {"starttime", "endtime"}.issubset(np_datetime_cols)

    def test_update_index_with_subpath_directory(self, ta_bank_no_index):
        """Ensure update index can have subpaths passed to it."""
        bank = ta_bank_no_index
        # get sub paths to excluse Z components
        sub_paths = "TA/M11A/VHE", "TA/M11A/VHN"
        bank.update_index(paths=sub_paths)
        # make sure only VHE and VHN are in index
        df = bank.read_index()
        assert set(df["channel"].unique()) == {"VHE", "VHN"}

    def test_update_index_with_supath_files(self, ta_bank_no_index):
        """Ensure a file (not just directory) can be used."""
        bank = ta_bank_no_index
        paths = [f"file_{x}.mseed" for x in range(1, 4)]
        # get sub paths to excluse Z components
        base_path = Path(bank.bank_path)
        # save a single new file in bank
        for tr, path in zip(obspy.read(), paths):
            # change origin time to avoid confusion with original files.
            tr.write(str(base_path / path), "mseed")
        # create paths mixing str, Path instance, relative, absolute
        in_paths = [base_path / paths[0], str(base_path / paths[1]), paths[2]]
        # update index, make sure everything is as expected
        df = bank.update_index(paths=in_paths).read_index()
        assert len(df) == 3

    def test_read_index_column_order(self, default_wbank):
        """The columns should be ordered according to WAVEFORM_DTYPES."""
        df = default_wbank.read_index()
        overlapping_cols = set(df.columns) & set(WAVEFORM_DTYPES)
        disjoint_cols = set(df.columns) - set(WAVEFORM_DTYPES)
        expected_order_1 = [x for x in WAVEFORM_DTYPES if x in overlapping_cols]
        expected_order_2 = sorted(disjoint_cols)
        expected_order = expected_order_1 + expected_order_2
        assert [str(x) for x in df.columns] == list(expected_order)

    def test_path_structure(self, tmpdir):
        """
        Ensure that it is possible to not use a path structure (see #178)
        """
        path = Path(tmpdir) / "path_structure"
        bank = WaveBank(path, path_structure="")
        assert bank.path_structure == ""

    def test_file_path_reconstruction(self, default_wbank):
        """
        It should be possible to get the full path of a file in the
        index using pathlib's "/" overloading
        """
        bank_path = default_wbank.bank_path
        index = default_wbank.read_index()
        pth = index.iloc[0].path
        assert (bank_path / pth).is_file()

    def test_file_path_legacy_index(self, default_wbank, legacy_path_index):
        """Verify backwards compatibility for relative paths with leading '/'"""
        st = default_wbank.get_waveforms()
        assert len(st)


class TestEmptyBank:
    """tests for graceful handling of empty WaveBanks"""

    @pytest.fixture()
    def empty_read_index_bank(self, empty_bank):
        """return the result of the empty read_index"""
        return empty_bank.read_index()

    @pytest.fixture()
    def empty_index_bank(self, empty_bank):
        """Write an unrelated table to index, return bank"""
        if empty_bank.index_path.exists():
            empty_bank.index_path.unlink()
        df = pd.DataFrame([1, 3, 4])
        with pd.HDFStore(empty_bank.index_path, "w") as store:
            store.put(key="weird", value=df)
        return empty_bank.update_index()

    # tests
    def test_empty_index_returned(self, empty_read_index_bank):
        """ensure an empty index (df) was returned"""
        assert isinstance(empty_read_index_bank, pd.DataFrame)
        assert empty_read_index_bank.empty
        assert set(empty_read_index_bank.columns).issuperset(WaveBank.index_columns)

    def test_metatable_exists(self, empty_index_bank):
        """Ensure the meta-table was added."""
        bank = empty_index_bank
        df = pd.read_hdf(bank.index_path, bank._meta_node)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


class TestReadIndex:
    """tests for getting the index"""

    # tests
    def test_attr(self, ta_bank_index):
        """test that the get index attribute exists"""
        assert hasattr(ta_bank_index, "read_index")

    def test_basic_filters1(self, ta_bank_index):
        """test exact matches work"""
        index = ta_bank_index.read_index(station="M11A")
        assert set(index.station) == {"M11A"}

    def test_basic_filters2(self, ta_bank_index):
        """that ? work in search"""
        index = ta_bank_index.read_index(station="M11?")
        assert set(index.station) == {"M11A"}

    def test_basic_filters3(self, ta_bank_index):
        """general wildcard searches work"""
        index = ta_bank_index.read_index(station="*")
        assert set(index.station) == {"M11A", "M14A"}

    def test_basic_filters4(self, ta_bank_index):
        """test combinations of wildcards work"""
        index = ta_bank_index.read_index(station="M*4?")
        assert set(index.station) == {"M14A"}

    def test_time_queries(self, ta_bank_index):
        """test that time queries return all data in range"""
        t1 = np.datetime64("2007-02-15T00:00:00")
        t2 = t1 + np.timedelta64(3600 * 60, "s")
        one_hour = np.timedelta64(1, "h")
        index = ta_bank_index.read_index(starttime=t1, endtime=t2)
        # ensure no data in the index fall far out of bounds of the query time
        # some may be slightly out of bounds due to the buffers used
        assert (index.endtime < (t1 - one_hour)).sum() == 0
        assert (index.starttime > (t2 + one_hour)).sum() == 0

    def test_time_queries2(self, ta_bank_index):
        """test that queries that are less than a file size work"""
        t1 = np.datetime64("2007-02-16T01:01:01")
        t2 = np.datetime64("2007-02-16T01:01:15")
        index = ta_bank_index.read_index(starttime=t1, endtime=t2)
        starttime = index.starttime.min()
        endtime = index.endtime.max()
        assert not index.empty
        assert t1 >= starttime
        assert t2 <= endtime

    def test_crandall_query(self, crandall_dataset):
        """
        Tests that querying the crandall_test dataset's event_waveforms.
        There was one event that didn't correctly get the waveforms
        on read index.
        """
        bank = crandall_dataset.waveform_client
        utc1 = obspy.UTCDateTime(2007, 8, 6, 10, 47, 25, 600_000)
        utc2 = obspy.UTCDateTime(2007, 8, 6, 10, 47, 25, 600_000)
        # this should have pulled 3 channels for each station
        df = bank.read_index(starttime=utc1, endtime=utc2)
        assert (df.groupby("station").size() == 3).all()

    def test_no_none_strs(self, ta_bank_index):
        """
        There shouldn't be any None strings in the df.
        These should have been replaced with proper None values.
        """
        df = ta_bank_index.read_index()
        assert "None" not in df.values

    def test_read_index_nullish_values(self, default_wbank):
        """Ensure Nullish Values return all streams"""
        bank = default_wbank
        df1 = bank.read_index(starttime=None, endtime=None)
        df2 = bank.read_index(starttime=pd.NaT, endtime=np.NaN)
        assert df1.equals(df2)

    def test_handles_pd_read_hdf_raises(self, default_wbank, monkeypatch):
        """Ensure when pd.read_hdf raises this is handled gracefully."""
        from tables.exceptions import ClosedNodeError

        state = {"has_raised": False}
        original_func = pd.read_hdf

        def raise_closed_node(*args, **kwargs):
            if state["has_raised"]:
                return original_func(*args, **kwargs)
            state["has_raised"] = True
            raise ClosedNodeError("simulated closed node")

        monkeypatch.setattr(pd, "read_hdf", raise_closed_node)

        default_wbank.update_index()
        df = default_wbank.read_index()
        assert not df.empty

    def test_read_index_raises_on_bad_param(self, default_wbank):
        """Ensure an unsupported kwarg raises. See #207."""
        with pytest.raises(UnsupportedKeyword, match="startime"):
            default_wbank.read_index(startime="2012-01-01")


class TestYieldStreams:
    """tests for yielding streams from the bank"""

    query1 = {
        "starttime": obspy.UTCDateTime("2007-02-15T00-00-10"),
        "endtime": obspy.UTCDateTime("2007-02-20T00-00-00"),
    }
    query2 = {
        "starttime": obspy.UTCDateTime("2007-02-15T00-00-10"),
        "endtime": obspy.UTCDateTime("2007-02-20T00-00-00"),
        "duration": 3600,
        "overlap": 60,
    }

    # fixtures
    @pytest.fixture(scope="function")
    def yield1(self, ta_bank_index):
        """the first yield set of parameters, duration not defined"""
        return ta_bank_index.yield_waveforms(**self.query1)

    @pytest.fixture(scope="function")
    def yield2(self, ta_bank_index):
        """
        The second yield set of parameters, duration and overlap are used.
        """
        return ta_bank_index.yield_waveforms(**self.query2)

    # tests
    def test_type(self, yield1):
        """test that calling yield_waveforms returns a generator"""
        assert isinstance(yield1, types.GeneratorType)
        count = 0
        for st in yield1:
            assert isinstance(st, obspy.Stream)
            assert len(st)
            count += 1
        assert count  # ensure some files were found

    def test_yield_with_durations(self, yield2, ta_index):
        """
        When durations is used each waveforms should have all the
        channels.
        """
        expected_stations = set(ta_index.station)
        t1 = self.query2["starttime"]
        dur = self.query2["duration"]
        overlap = self.query2["overlap"]
        for st in yield2:
            stations = set([x.stats.station for x in st])
            assert stations == expected_stations
            # check start and endtimes
            assert all([abs(x.stats.starttime - t1) < 2.0 for x in st])
            t2 = t1 + dur + overlap
            if t2 >= self.query2["endtime"] + overlap:
                t2 = self.query2["endtime"] + overlap
            assert all([abs(x.stats.endtime - t2) < 2.0 for x in st])
            t1 += dur


class TestGetWaveforms:
    """tests for getting waveforms from the index"""

    # fixture params
    query1 = {
        "starttime": obspy.UTCDateTime("2007-02-15T00-00-10"),
        "endtime": obspy.UTCDateTime("2007-02-20T00-00-00"),
        "station": "*",
        "network": "*",
        "channel": "VHE",
    }

    query2 = {
        "starttime": obspy.UTCDateTime("2007-02-15T00-00-10"),
        "endtime": obspy.UTCDateTime("2007-02-20T00-00-00"),
        "station": "*",
        "network": "*",
        "channel": "VH[NE]",
    }

    query3 = {
        "station": ["SPS", "WTU", "CFS"],
        "channel": ["HHZ", "ENZ"],
        "starttime": obspy.UTCDateTime("2013-04-11T05:07:26.330000Z"),
        "endtime": obspy.UTCDateTime("2013-04-11T05:08:26.328000Z"),
    }

    # fixtures
    @pytest.fixture(scope="class")
    def stream1(self, ta_bank):
        """return the waveforms using query1 as params"""
        out = ta_bank.get_waveforms(**self.query1)
        return out

    @pytest.fixture(scope="class")
    def stream2(self, ta_bank):
        """return the waveforms using query2 as params"""
        return ta_bank.get_waveforms(**self.query2)

    @pytest.fixture(scope="class")
    def stream3(self, bingham_dataset):
        """return a waveforms from query params 3 on bingham_test dataset"""
        bank = bingham_dataset.waveform_client
        return bank.get_waveforms(**self.query3)

    @pytest.fixture
    def bank49(self, tmpdir):
        """setup a WaveBank to test issue #49."""
        path = Path(tmpdir)
        # create two traces with a slight gap between the two
        st1 = obspy.read()
        st2 = obspy.read()
        for tr1, tr2 in zip(st1, st2):
            tr1.stats.starttime = tr1.stats.endtime + 10
        # save files to directory, create bank and update
        st1.write(str(path / "st1.mseed"), "mseed")
        st2.write(str(path / "st2.mseed"), "mseed")
        bank = obsplus.WaveBank(path)
        bank.update_index()
        return bank

    @pytest.fixture
    def bank_null_loc_codes(self, tmpdir):
        """create a bank that has nullish location codes in its streams."""
        st = obspy.read()
        path = Path(tmpdir)
        for tr in st:
            tr.stats.location = "--"
            time = str(get_reference_time(tr))
            name = time.split(".")[0].replace(":", "-") + f"_{tr.id}"
            tr.write(str(path / name) + ".mseed", "mseed")
        bank = WaveBank(path)
        bank.update_index()
        return bank

    # tests
    def test_attr(self, ta_bank_index):
        """test that the bank class has the get_waveforms attr"""
        assert hasattr(ta_bank_index, "get_waveforms")

    def test_stream1(self, stream1):
        """make sure stream1 has all the expected features"""
        assert isinstance(stream1, obspy.Stream)
        assert len(stream1) == 2
        # get some stats from waveforms
        channels = {tr.stats.channel for tr in stream1}
        starttime = min([tr.stats.starttime for tr in stream1])
        endtime = max([tr.stats.endtime for tr in stream1])
        assert len(channels) == 1
        assert abs(starttime - self.query1["starttime"]) <= 1.0
        assert abs(endtime - self.query1["endtime"]) <= 1.0

    def test_bracket_matches(self, stream2):
        """
        Make sure the bracket style filters work (eg VH[NE] for both
        VHN and VHE.
        """
        channels = {tr.stats.channel for tr in stream2}
        assert channels == {"VHN", "VHE"}

    def test_filter_with_multiple_trace_files(self, crandall_bank):
        """Ensure a bank with can be correctly filtered."""
        t1 = obspy.UTCDateTime("2007-08-06T01-44-48")
        t2 = t1 + 60
        st = crandall_bank.get_waveforms(
            starttime=t1, endtime=t2, network="TA", channel="BHZ"
        )
        assert len(st)
        for tr in st:
            assert tr.stats.network == "TA"
            assert tr.stats.channel == "BHZ"

    def test_list_params(self, stream3):
        """ensure parameters can be passed as lists"""
        # get a list of seed ids and ensure they are the same in the query
        sids = {tr.id for tr in stream3}
        for sid in sids:
            for key, val in zip(NSLC, sid.split(".")):
                sequence = self.query3.get(key)
                if sequence is not None:
                    assert val in sequence

    def test_issue_49(self, bank49):
        """
        Ensure traces with masked arrays are not returned by get_waveforms.
        """
        st = bank49.get_waveforms()
        for tr in st:
            assert not isinstance(tr.data, np.ma.MaskedArray)
        assert len(st.get_gaps()) == 3

    def test_stream_null_location_codes(self, bank_null_loc_codes):
        """
        Ensure bank still works when stations have nullish location codes.
        """
        bank = bank_null_loc_codes
        df = bank.read_index()
        assert len(df) == 3
        st = bank.get_waveforms()
        assert len(st) == 3

    def test_de_duplicate_stream(self, duplicate_bank):
        """get_waveforms should de-dup/merge when calling get_waveforms."""
        std = obspy.read()  # default
        st = duplicate_bank.get_waveforms()
        assert len(st) == 3, "traces were not merged!"
        starttime = min([tr.stats.starttime for tr in st])
        endtime = max([tr.stats.endtime for tr in st])
        default_duration = std[0].stats.endtime - std[0].stats.starttime
        assert (endtime - starttime) > default_duration


class TestGetBulkWaveforms:
    """tests for pulling multiple waveforms using get_bulk_waveforms"""

    t1, t2 = obspy.UTCDateTime("2007-02-16"), obspy.UTCDateTime("2007-02-18")
    bulk1 = [("TA", "M11A", "*", "*", t1, t2), ("TA", "M14A", "*", "*", t1, t2)]
    standard_query1 = {"station": "M1?A", "starttime": t1, "endtime": t2}

    # fixtures
    @pytest.fixture(scope="class")
    def ta_bulk_1(self, ta_bank):
        """perform the first bulk query and return the result"""
        return strip_processing(ta_bank.get_waveforms_bulk(self.bulk1))

    @pytest.fixture(scope="class")
    def ta_standard_1(self, ta_bank):
        """perform the standard query, return the result"""
        st = ta_bank.get_waveforms(**self.standard_query1)
        return strip_processing(st)

    @pytest.fixture
    def bank_3(self, tmpdir_factory):
        """Create a bank with several different types of streams."""
        td = tmpdir_factory.mktemp("waveforms")
        t1, t2 = self.t1, self.t2
        bulk3 = [
            ("TA", "M11A", "01", "CHZ", t1, t2),
            ("RR", "BOB", "", "HHZ", t1, t2),
            ("BB", "BOB", "02", "ENZ", t1, t2),
            ("UU", "SRU", "--", "HHN", t1, t2),
        ]
        ArchiveDirectory(str(td)).create_directory_from_bulk_args(bulk3)
        bank = WaveBank(str(td))
        bank.update_index()
        return bank

    # tests
    def test_equal_results(self, ta_bulk_1, ta_standard_1):
        """the two queries should return the same streams"""
        assert ta_bulk_1 == ta_standard_1

    def test_bulk3_no_matches(self, bank_3):
        """tests for bizarre wildcard usage. Should not return no data."""
        bulk = [
            ("TA", "*", "*", "HHZ", self.t1, self.t2),
            ("*", "MOB", "*", "*", self.t1, self.t2),
            ("BB", "BOB", "1?", "*", self.t1, self.t2),
        ]
        st = bank_3.get_waveforms_bulk(bulk)
        assert len(st) == 0, "no waveforms should have been returned!"

    def test_bulk3_one_match(self, bank_3):
        """Another bulk request that should return one trace."""
        bulk = [
            ("TA", "*", "12", "???", self.t1, self.t2),
            ("*", "*", "*", "CHZ", self.t1, self.t2),
        ]
        st = bank_3.get_waveforms_bulk(bulk)
        assert len(st) == 1

    def test_no_matches(self, ta_bank):
        """Test waveform bulk when no params meet req."""
        t1 = obspy.UTCDateTime("2012-01-01")
        t2 = t1 + 12
        bulk = [("bob", "is", "no", "sta", t1, t2)]
        stt = ta_bank.get_waveforms_bulk(bulk)
        assert isinstance(stt, obspy.Stream)

    def test_empty_bank(self, empty_bank):
        """Test waveform bulk when no params meet req."""
        t1 = obspy.UTCDateTime("2012-01-01")
        t2 = t1 + 12
        bulk = [("bob", "is", "no", "sta", t1, t2)]
        stt = empty_bank.get_waveforms_bulk(bulk)
        assert isinstance(stt, obspy.Stream)

    def test_one_match(self, ta_bank):
        """Test waveform bulk when there is one req. that matches"""
        df = ta_bank.read_index()
        row = df.iloc[0]
        nslc = [getattr(row, x) for x in NSLC] + [row.starttime, row.endtime]
        bulk = [tuple(nslc)]
        stt = ta_bank.get_waveforms_bulk(bulk)
        assert isinstance(stt, obspy.Stream)

    def test_no_duplicate_traces(self, duplicate_bank):
        """Ensure overlap in bulk args doesn't result in overlapping traces."""
        st = duplicate_bank.get_waveforms()
        args = list(NSLC) + ["starttime", "endtime"]
        bulk = [{i: tr.stats[i] for i in args} for tr in st]
        # create overlaps
        new = copy.deepcopy(bulk)
        for new_b in new:
            new_b["starttime"] += 5
            new_b["endtime"] -= 5
        # and duplicates
        bulky_bulk = bulk + bulk + new

        st = duplicate_bank.get_waveforms_bulk(bulky_bulk)
        assert len(st) == 3


class TestBankCache:
    """test that the time cache avoids repetitive queries to the h5 index"""

    query_1 = TestGetBulkWaveforms.standard_query1

    # fixtures
    @pytest.fixture(scope="class")
    def mp_ta_bank(self, ta_bank):
        """monkey patch the ta_bank instance to count how many times the .h5
        index is accessed, store it on the accessed_times in the ._cache.times
        """
        func = count_calls(ta_bank, ta_bank._index_cache._get_index, "index_calls")
        ta_bank._index_cache._get_index = func
        return ta_bank

    @pytest.fixture(scope="class")
    def query_twice_mp_ta(self, mp_ta_bank):
        """query the instrumented ta_test bank twice"""
        _ = mp_ta_bank.read_index(**self.query_1)
        _ = mp_ta_bank.read_index(**self.query_1)
        return mp_ta_bank

    # tests
    def test_query_twice(self, query_twice_mp_ta):
        """make sure a double query only accesses index once"""
        assert query_twice_mp_ta.index_calls == 1


class TestBankCacheWithKwargs:
    """
    kwargs should get hashed as well, so the same times with different
    kwargs should be cached.
    """

    @pytest.fixture
    def got_gaps_bank(self, ta_bank):
        """call get_gaps on bank (to populate cache) and return"""
        ta_bank.get_gaps_df()
        return ta_bank

    def test_get_gaps_doesnt_overwrite_cache(self, got_gaps_bank):
        """
        Ensure that calling get gaps doesn't result in read_index
        not returning the path column.
        """
        inds = got_gaps_bank.read_index()
        assert "path" in inds.columns


class TestPutWaveforms:
    """test that waveforms can be put into the bank"""

    # fixtures
    @pytest.fixture(scope="class")
    def add_stream(self, ta_bank):
        """add the default obspy waveforms to the bank, return the bank"""
        st = obspy.read()
        ta_bank.update_index()  # make sure index cache is set
        ta_bank.put_waveforms(st, update_index=True)
        return ta_bank

    @pytest.fixture(scope="class")
    def default_stations(self):
        """return the default stations on the default waveforms as a set"""
        return set([x.stats.station for x in obspy.read()])

    @pytest.fixture(scope="class")
    def gappy_default_stream(self):
        """Create a gappy stream from default stream."""
        st = obspy.read()
        start, end = st[0].stats.starttime, st[0].stats.endtime
        duration = end - start
        mid = start + duration / 2.0
        tr1 = st[0].slice(starttime=start, endtime=mid - 1 / 100.0)
        tr2 = st[0].slice(starttime=mid + 1 / 100.0, endtime=end)
        st[0] = tr1 + tr2
        return st

    # tests
    def test_deposited_waveform(self, add_stream, default_stations):
        """make sure the waveform was added to the bank"""
        assert default_stations.issubset(add_stream.read_index().station)

    def test_retrieve_stream(self, add_stream):
        """ensure the default waveforms can be pulled out of the archive"""
        st1 = add_stream.get_waveforms(station="RJOB").sort()
        st2 = obspy.read().sort()
        assert len(st1) == 3
        for tr1, tr2 in zip(st1, st2):
            assert np.all(tr1.data == tr2.data)

    def test_put_waveforms_to_crandall_copy(self, tmpdir):
        """
        Ran into issue in docs where putting data into the crandall_test
        copy didn't work.
        """
        ds = obsplus.utils.dataset.copy_dataset(
            dataset="crandall_test", destination=Path(tmpdir)
        )
        bank = WaveBank(ds.waveform_client)
        ind1 = bank.read_index()  # this sets cache
        # ensure RJOB is not yet in the bank
        assert "RJOB" not in set(ind1["station"].unique())
        st = obspy.read()
        bank.put_waveforms(st, update_index=True)
        df = bank.read_index(station="RJOB")
        assert len(df) == len(st)
        assert set(df.station) == {"RJOB"}

    def test_put_gappy_data(self, default_wbank, gappy_default_stream):
        """Tests for putting a stream with gaps into a wavebank."""
        # test succeeds if this doesn't fail
        default_wbank.put_waveforms(gappy_default_stream)


class TestPutMultipleTracesOneFile:
    """ensure that multiple waveforms can be put into one file"""

    st = obspy.read()
    st_mod = st.copy()
    for tr in st_mod:
        tr.stats.station = "PS"
    expected_seeds = {tr.id for tr in st + st_mod}

    # fixtures
    @pytest.fixture(scope="class")
    def bank(self):
        """return an empty bank for depositing waveforms"""
        bd = dict(path_structure="streams/network", name_structure="time")
        with tempfile.TemporaryDirectory() as tempdir:
            out = os.path.join(tempdir, "temp")
            yield WaveBank(out, **bd)

    @pytest.fixture(scope="class")
    def deposited_bank(self, bank: obsplus.WaveBank):
        """deposit the waveforms in the bank, return the bank"""
        bank.put_waveforms(self.st_mod, update_index=True)
        bank.put_waveforms(self.st, update_index=True)
        return bank

    @pytest.fixture(scope="class")
    def mseed_files(self, deposited_bank):
        """count the number of files"""
        bfile = deposited_bank.bank_path
        glo = glob.glob(os.path.join(bfile, "**", "*.mseed"), recursive=True)
        return glo

    @pytest.fixture(scope="class")
    def banked_stream(self, mseed_files):
        """Return a stream of all the bank contents."""
        st = obspy.Stream()
        for ftr in mseed_files:
            st += obspy.read(ftr)
        return st

    @pytest.fixture(scope="class")
    def number_of_files(self, mseed_files):
        """count the number of files"""
        return len(mseed_files)

    # tests
    def test_one_file(self, number_of_files):
        """ensure only one file was written"""
        assert number_of_files == 1

    def test_all_streams(self, banked_stream):
        """
        Ensure all the channels in the waveforms where written to
        the bank.
        """
        banked_seed_ids = {tr.id for tr in banked_stream}
        assert banked_seed_ids == self.expected_seeds


class TestBadWaveforms:
    """test how wavebank handles bad waveforms"""

    # fixtures
    @pytest.fixture(scope="class")
    def ta_bank_bad_file(self, ta_bank):
        """add an unreadable file to the wave bank, then return new bank"""
        path = ta_bank.bank_path
        new_file_path = os.path.join(path, "bad_file.mseed")
        with open(new_file_path, "w") as fi:
            fi.write("this is not an mseed file, duh")
        # remove old index if it exists
        if os.path.exists(ta_bank.index_path):
            os.remove(ta_bank.index_path)
        return WaveBank(path)

    @pytest.fixture
    def ta_bank_empty_files(self, ta_bank, tmpdir):
        """add many empty files to bank, ensure index still reads all files"""
        old_path = Path(ta_bank.bank_path)
        new_path = Path(tmpdir) / "waveforms"
        shutil.copytree(old_path, new_path)
        # create 100 empty files
        for a in range(100):
            new_file_path = new_path / f"{a}.mseed"
            with new_file_path.open("wb"):
                pass
        # remove old index if it exists
        index_path = new_path / (Path(ta_bank.index_path).name)
        if index_path.exists():
            os.remove(index_path)
        bank = WaveBank(old_path)
        bank.update_index()
        return bank

    # tests
    def test_bad_file_emits_warning(self, ta_bank_bad_file):
        """ensure an unreadable waveform file will emit a warning"""

        with pytest.warns(UserWarning) as record:
            ta_bank_bad_file.update_index()
        assert len(record)
        expected_str = "obspy failed to read"
        assert any([expected_str in r.message.args[0] for r in record])

    def test_read_index(self, ta_bank_empty_files, ta_bank):
        """tests for bank with many empty files"""
        df1 = ta_bank_empty_files.read_index()
        df2 = ta_bank.read_index()
        assert (df1 == df2).all().all()

    def test_get_non_existent_waveform(self, ta_bank):
        """Ensure asking for a non-existent station returns empty waveforms."""
        st = ta_bank.get_waveforms(station="RJOB")
        assert isinstance(st, obspy.Stream)
        assert len(st) == 0


class TestFilesWithMultipleChannels:
    """make sure banks that have multi-channel files (eg events) behave"""

    counter = 0

    # fixtures
    @pytest.fixture(autouse=True, scope="class")
    def count_read_executions(self, bank):
        """patch read to make sure it is called correct number of times"""

        def count_decorator(func):
            def wrapper(*args, **kwargs):
                self.counter += 1
                return func(*args, **kwargs)

            return wrapper

        # update index first so we only count event reads
        bank.update_index()

        old_func = obsplus.utils.misc.READ_DICT["mseed"]
        new_func = count_decorator(old_func)
        obsplus.utils.misc.READ_DICT["mseed"] = new_func
        yield
        self.counter = 0
        obsplus.utils.misc.READ_DICT["mseed"] = old_func

    @pytest.fixture(scope="class")
    def multichannel_bank(self):
        """return a directory with a mseed that has multiple channels"""
        st = obspy.read()
        with tempfile.TemporaryDirectory() as tdir:
            path = join(tdir, "test.mseed")
            st.write(path, "mseed")
            yield tdir
        if os.path.exists(tdir):
            shutil.rmtree(tdir)

    @pytest.fixture(scope="class")
    def bank(self, multichannel_bank):
        """return a wavefetcher using multichannel bank"""
        return WaveBank(multichannel_bank)

    @pytest.fixture(scope="class")
    def bulk_args(self):
        """return bulk args for default waveforms"""
        st = obspy.read()
        out = []
        for tr in st:
            net, sta = tr.stats.network, tr.stats.station
            loc, cha = tr.stats.location, tr.stats.channel
            t1, t2 = tr.stats.starttime, tr.stats.endtime
            out.append((net, sta, loc, cha, t1, t2))
        return out

    @pytest.fixture(scope="class")
    def bulk_st(self, bank, bulk_args):
        """return the result of getting bulk args"""
        return bank.get_waveforms_bulk(bulk_args)

    @pytest.fixture(scope="class")
    def number_of_calls(self, bulk_st):
        """return the nubmer of calls made to read"""
        return self.counter

    # tests
    def test_stream_len(self, bulk_st):
        """ensure exactly 3 channels are in waveforms"""
        assert len(bulk_st) == 3

    def test_read_stream_called_once(self, number_of_calls):
        """assert the read function was called exactly once"""
        assert number_of_calls == 1


class TestGetAvailability:
    """test that WaveBank will return an availability dataframe"""

    # fixtures
    @pytest.fixture(scope="class")
    def avail_df(self, ta_bank):
        """return the availability dataframe"""
        return ta_bank.get_availability_df()

    @pytest.fixture(scope="class")
    def avail(self, ta_bank):
        """return availability"""
        return ta_bank.availability()

    # test
    def test_availability_df(self, avail_df):
        """test the availability property returns a dataframe"""
        assert isinstance(avail_df, pd.DataFrame)
        assert not avail_df.empty

    def test_avail_df_filter(self, ta_bank):
        """ensure specifying a network/station argument filters df"""
        df = ta_bank.get_availability_df(station="M14*", channel="*Z")
        assert len(df) == 1
        assert df.iloc[0].station == "M14A"
        assert df.iloc[0].channel == "VHZ"

    def test_availability(self, avail):
        """Test output shape of availability."""
        for av in avail:
            assert len(av) == 6
            # first four values should be strings
            for val in av[:4]:
                assert isinstance(val, str)
            # last two values should be UTCDateTimes
            for val in av[4:]:
                assert isinstance(val, obspy.UTCDateTime)
            assert isinstance(av[0], str)


class TestGetGaps:
    """test that the get_gaps method returns info about gaps"""

    start = UTC("2017-09-18")
    end = UTC("2017-09-28")
    sampling_rate = 1

    gaps = [
        (UTC("2017-09-18T18-00-00"), UTC("2017-09-18T19-00-00")),
        (UTC("2017-09-18T20-00-00"), UTC("2017-09-18T20-00-15")),
        (UTC("2017-09-20T01-25-35"), UTC("2017-09-20T01-25-40")),
        (UTC("2017-09-21T05-25-35"), UTC("2017-09-25T10-36-42")),
    ]

    durations = np.array([y - x for x, y in gaps])

    durations_timedelta = np.array([to_timedelta64(float(x)) for x in durations])

    overlap = 0

    def _make_gappy_archive(self, path):
        """Create the gappy archive defined by params in class."""
        ArchiveDirectory(
            path,
            self.start,
            self.end,
            self.sampling_rate,
            gaps=self.gaps,
            overlap=self.overlap,
        ).create_directory()
        return path

    # fixtures
    @pytest.fixture(scope="class")
    def gappy_dir(self, class_tmp_dir):
        """create a directory that has gaps in it"""
        self._make_gappy_archive(join(class_tmp_dir, "temp1"))
        return class_tmp_dir

    @pytest.fixture(scope="class")
    def gappy_bank(self, gappy_dir):
        """init a WaveBank on the gappy data"""
        bank = WaveBank(gappy_dir)
        # make sure index is updated after gaps are introduced
        if os.path.exists(bank.index_path):
            os.remove(bank.index_path)
        bank.update_index()
        return bank

    @pytest.fixture()
    def gappy_and_contiguous_bank(self, tmp_path):
        """Create a directory with gaps and continuous data"""
        # first create directory with gaps
        self._make_gappy_archive(tmp_path)
        # first write data with no gaps
        st = obspy.read()
        for num, tr in enumerate(st):
            tr.stats.station = "GOOD"
            tr.write(str(tmp_path / f"good_{num}.mseed"), "mseed")
        return WaveBank(tmp_path).update_index()

    @pytest.fixture(scope="class")
    def empty_bank(self):
        """create a bank object initiated on an empty directory"""
        with tempfile.TemporaryDirectory() as td:
            bank = WaveBank(td)
            yield bank

    @pytest.fixture(scope="class")
    def gap_df(self, gappy_bank):
        """return a gap df from the gappy bank"""
        return gappy_bank.get_gaps_df()

    @pytest.fixture(scope="class")
    def uptime_df(self, gappy_bank):
        """return the uptime dataframe from the gappy bank"""
        return gappy_bank.get_uptime_df()

    @pytest.fixture(scope="class")
    def segment_df(self, gappy_bank):
        """Return the segment dataframe from the gappy bank"""
        return gappy_bank.get_segments_df()

    @pytest.fixture()
    def uptime_default(self, default_wbank):
        """return the uptime from the default stream bank."""
        return default_wbank.get_uptime_df()

    @pytest.fixture()
    def segment_default(self, default_wbank):
        """Return the segment dataframe from the default stream bank"""
        return default_wbank.get_segments_df()

    @pytest.fixture()
    def small_overlap_gaps(self, tmpdir):
        """
        Create a bank with small overlapping files.
        """

        def create_trace(row):
            """Create a trace in the middle of a row of the index."""
            t1 = to_utc(row["starttime"]) + 10
            data = np.random.rand(10)
            header = dict(starttime=t1, sampling_rate=1)
            for code in NSLC:
                header[code] = row[code]
            return obspy.Trace(data, header=header)

        t1, t2 = obspy.UTCDateTime("2017-01-01"), obspy.UTCDateTime("2017-01-02")
        sid = ("TA.BOB.01.VHZ",)
        kwargs = dict(starttime=t1, endtime=t2, path=tmpdir, seed_ids=sid)
        ArchiveDirectory(**kwargs).create_directory()
        bank = obsplus.WaveBank(tmpdir).update_index()
        index = bank.read_index()
        # create a trace and push into bank
        tr = create_trace(index.sort_values("starttime").iloc[4])
        bank.put_waveforms(tr, update_index=True)
        assert len(bank.read_index()) == len(index) + 1, "one trace added"
        return bank

    # tests
    def test_gaps_length(self, gap_df):  # , gappy_bank):
        """ensure each of the gaps shows up in df"""
        assert isinstance(gap_df, pd.DataFrame)
        assert not gap_df.empty
        group = gap_df.groupby(["network", "station", "location", "channel"])
        sampling_period = gap_df["sampling_period"].iloc[0]
        for gnum, df in group:
            assert len(df) == len(self.gaps)
            dif = abs(df["gap_duration"] - self.durations_timedelta)
            assert (dif < (1.5 * sampling_period)).all()

    def test_gappy_uptime_df(self, uptime_df):
        """ensure the uptime df is of correct type and accurate"""
        assert isinstance(uptime_df, pd.DataFrame)
        gap_duration = sum([x[1] - x[0] for x in self.gaps])
        duration = self.end - self.start
        uptime_percent = (duration - gap_duration) / duration
        np_assert(uptime_df["availability"], uptime_percent, atol=0.001)

    def test_gappy_segment_df(self, segment_df):
        """ensure the segment df is the correct type and accurate"""
        assert isinstance(segment_df, pd.DataFrame)
        # There should be one entry for each continuous data segment
        #  (will be 2 longer than the corresponding gap df)
        #  Also, the gaps are at the same times for both channels
        assert len(segment_df) - 2 == len(self.gaps) * 2
        # There should be a predictable amount of data
        gap_duration = sum([x[1] - x[0] for x in self.gaps])
        duration = self.end - self.start
        # Explanation of the corrections to uptime:
        #  There are two channels in the archive and each has a gap, so things
        #   need to be doubled
        #  The uptimes are going to be 1 sample shorter than what is calculated
        #   due to not including the starttime of the gap
        #  There is one more entry for each channel than there is gaps
        uptime = (
            duration - gap_duration - (len(self.gaps) + 1) * 1 / self.sampling_rate
        ) * 2
        np_assert(
            segment_df["duration"].dt.total_seconds().sum(),
            uptime,
            atol=1 / self.sampling_rate,
        )

    def test_uptime_default(self, uptime_default):
        """
        Ensure the uptime of the basic bank (no gaps) has expected times/channels.
        """
        df = uptime_default
        st = obspy.read()
        assert not df.empty, "uptime df is empty"
        assert len(df) == len(st)
        assert {tr.id for tr in st} == set(obsplus.utils.pd.get_seed_id_series(df))
        assert (df["gap_duration"] == EMPTYTD64).all()

    def test_segments_default(self, segment_default):
        """Ensure segment df of bank w/o gaps has expected info"""
        # Should essentially match the summary uptime dataframe
        st = obspy.read()
        assert len(segment_default) == len(st)
        assert {tr.id for tr in st} == set(
            obsplus.utils.pd.get_seed_id_series(segment_default)
        )

    def test_empty_directory(self, empty_bank):
        """
        Ensure an empty bank get_gaps returns and empty df with expected
        columns.
        """
        gaps = empty_bank.get_gaps_df()
        assert not len(gaps)
        assert set(WaveBank._gap_columns).issubset(set(gaps.columns))

    def test_empty_directory_segments(self, empty_bank):
        """
        Ensure empty bank returns an empty df with expected columns
        """
        segments = empty_bank.get_segments_df(detailed=True)
        assert not len(segments)
        assert set(segments.columns) == set(empty_bank._segment_columns)

    def test_ta_uptime(self, ta_dataset):
        """ensure the ta bank returns an uptime df"""
        bank = ta_dataset.waveform_client
        df = bank.get_uptime_df()
        diff = abs(df["uptime"] - df["duration"])
        tolerance = np.timedelta64(1, "s")
        assert (diff < tolerance).all()

    def test_gappy_and_contiguous_uptime(self, gappy_and_contiguous_bank):
        """
        Ensure when there are gappy streams and contiguous streams
        get_uptime still returns correct results.
        """
        wbank = gappy_and_contiguous_bank
        index = wbank.read_index()
        uptime = wbank.get_uptime_df()
        # make sure the same seed ids are in the index as uptime df
        seeds_from_index = set(obsplus.utils.pd.get_seed_id_series(index))
        seeds_from_uptime = set(obsplus.utils.pd.get_seed_id_series(uptime))
        assert seeds_from_index == seeds_from_uptime
        assert not uptime.isnull().any().any()

    def test_gappy_and_contiguous_segments(self, gappy_and_contiguous_bank):
        """
        Ensure a combo of gappy and contiguous streams returns
        the correct result
        """
        index = gappy_and_contiguous_bank.read_index()
        uptime = gappy_and_contiguous_bank.get_uptime_df()
        segments = gappy_and_contiguous_bank.get_segments_df()
        # More records than default uptime summary, but condenses contents
        #  of the index
        assert len(uptime) < len(segments) < len(index)

    def test_no_gaps_on_continuous_dataset(self, ta_dataset):
        """test no gaps on ta dataset."""
        ds = ta_dataset
        wbank = ds.waveform_client
        gap_df = wbank.get_gaps_df()
        assert len(gap_df) == 0

    def test_no_segment_gaps_on_continuous_dataset(self, ta_dataset):
        """Segments should match availability df"""
        wbank = ta_dataset.waveform_client
        avail = wbank.get_availability_df()
        segments = wbank.get_segments_df()
        assert len(segments) == len(avail)
        seeds_from_uptime = set(obsplus.utils.pd.get_seed_id_series(avail))
        seeds_from_detailed = set(obsplus.utils.pd.get_seed_id_series(segments))
        assert seeds_from_detailed == seeds_from_uptime
        assert (segments.starttime == avail.starttime).all()
        assert (segments.endtime == avail.endtime).all()

    def test_gaps_small_overlaps(self, small_overlap_gaps):
        """
        Ensure when there are files with small overlaps gaps are not falsely
        reported.
        """
        gap_df = small_overlap_gaps.get_gaps_df()
        assert len(gap_df) == 0

    def test_segments_small_overlaps(self, small_overlap_gaps):
        """Make sure small overlaps are handled correctly"""
        avail = small_overlap_gaps.get_availability_df()
        segments = small_overlap_gaps.get_segments_df()
        # Once again should match the uptime df
        assert len(segments) == len(avail)
        seeds_from_uptime = set(obsplus.utils.pd.get_seed_id_series(avail))
        seeds_from_detailed = set(obsplus.utils.pd.get_seed_id_series(segments))
        assert seeds_from_detailed == seeds_from_uptime
        assert (segments.starttime == avail.starttime).all()
        assert (segments.endtime == avail.endtime).all()

    def test_min_gap_param(self, gappy_bank):
        """Ensure the min gap parameter works."""
        no_gap = gappy_bank.get_gaps_df(min_gap=10000000)
        with_gap = gappy_bank.get_gaps_df()
        assert len(no_gap) < len(with_gap)

    def test_segments_min_gap_param(self, gappy_bank):
        """Ensure the min gap parameter works for get_segments_df"""
        wo_min_gap = gappy_bank.get_segments_df()
        w_min_gap = gappy_bank.get_segments_df(min_gap=10000000)
        assert len(w_min_gap) < len(wo_min_gap)


class TestBadInputs:
    """ensure wavebank handles bad inputs correctly"""

    # tests
    def test_bad_inventory(self, tmp_ta_dir):
        """ensure giving a bad stations str raises"""
        with pytest.raises(Exception):
            WaveBank(tmp_ta_dir, inventory="some none existent file")


class TestConcurrentReads:
    """
    Tests for concurrent reads.
    """

    @pytest.fixture
    def wbank_executor(self, ta_bank, monkeypatch, instrumented_executor):
        """Return a wavebank with an instrumented executor."""
        monkeypatch.setattr(ta_bank, "executor", instrumented_executor)
        with suppress(FileNotFoundError):
            os.remove(ta_bank.index_path)
        return ta_bank

    def test_concurrent_get_waveforms(self, wbank_executor):
        """Ensure conccurent get_waveforms uses executor."""
        _ = wbank_executor.get_waveforms()
        counter = getattr(wbank_executor.executor, "_counter", {})
        assert counter.get("map", 0) > 0

    def test_concurrent_update_index(self, wbank_executor):
        """Ensure updating index can be performed with executor."""
        _ = wbank_executor.update_index()
        counter = getattr(wbank_executor.executor, "_counter", {})
        assert counter.get("map", 0) > 0


class TestConcurrentBank:
    """
    Concurrency tests for bank operations which can be run in multiple
    threads/processes.
    """

    worker_count = 3
    new_files = 1

    def func(self, wbank):
        """add new files to the wavebank then update index, return index."""
        try:
            wbank.read_index()
        except Exception as e:
            return str(e)

    # fixtures
    @pytest.fixture
    def concurrent_bank(self, tmpdir):
        """Make a temporary bank and index it."""
        st = obspy.read()
        st.write(str(Path(tmpdir) / "test.mseed"), "mseed")
        wbank = WaveBank(str(tmpdir)).update_index()
        self.func(wbank)
        return wbank

    @pytest.fixture
    def thread_read(self, concurrent_bank, thread_executor):
        """
        Run a bunch of update index operations in different threads,
        return list of results.
        """
        out = []
        concurrent_bank.update_index()
        func = functools.partial(self.func, wbank=concurrent_bank)
        for _ in range(self.worker_count):
            out.append(thread_executor.submit(func))
        return list(as_completed(out))

    @pytest.fixture(scope="class")
    def process_pool(self):
        """return a thread pool"""
        with ProcessPoolExecutor(self.worker_count) as executor:
            yield executor

    @pytest.fixture
    def process_read(self, concurrent_bank, process_pool):
        """
        Run a bunch of update index operations in different processes,
        return list of results.
        """
        concurrent_bank.update_index()
        out = []
        func = functools.partial(self.func, wbank=concurrent_bank)
        for num in range(self.worker_count):
            time.sleep(0.1)
            out.append(process_pool.submit(func))
        return list(as_completed(out))

    # tests
    def test_pickle_bank(self, concurrent_bank):
        """Ensure the bank can be pickled."""
        pkl = pickle.dumps(concurrent_bank)
        new_bank = pickle.loads(pkl)
        assert isinstance(new_bank, WaveBank)
        assert new_bank.bank_path == concurrent_bank.bank_path

    def test_index_read_thread(self, thread_read):
        """
        Ensure the index updated and the threads didn't kill each other.
        """
        # get a list of exceptions that occurred
        assert len(thread_read) == self.worker_count
        excs = [x.result() for x in thread_read]
        excs = [x for x in excs if x is not None]
        if excs:
            msg = f"Exceptions were raised by the thread pool:\n {excs}"
            pytest.fail(msg)

    def test_index_read_process(self, process_read):
        """
        Ensure the index can be updated in different processes.
        """
        assert len(process_read) == self.worker_count
        # ensure no exceptions were raised
        excs = [x.result() for x in process_read]
        excs = [x for x in excs if x is not None]
        if excs:
            msg = f"Exceptions were raised by the process pool:\n {excs}"
            pytest.fail(msg)


class TestSelectDoesntReturnSuperset:
    """make sure selecting on an attribute doesnt return a superset of that
    attribute. EG, selecting station '2' should not also return station '22'
    """

    # fixtures
    @pytest.fixture(scope="class")
    def df_index(self):
        """return a dataframe index for testing"""
        stations = [str(x) for x in [2, 2, 2, 22, 22, 22, 222, 222, 222]]
        df = pd.DataFrame(index=range(9), columns=WaveBank.index_columns)
        df["station"] = stations
        df["network"] = "1"
        df["starttime"] = np.datetime64("now")
        df["endtime"] = df["starttime"] + np.timedelta64(3600, "s")
        df["sampling_period"] = 0.01
        df["path"] = "abcd"
        return df

    @pytest.fixture(scope="class")
    def modified_index_wbank(self, df_index, ta_archive, tmp_path_factory):
        """return a wbank with station names that have been modified"""
        wbank = WaveBank(ta_archive)
        # Point the bank path to a temporary directory
        path = tmp_path_factory.mktemp("temp_waveforms")
        wbank.bank_path = path
        # Make the bank write out the modified index
        wbank._write_update(df_index, time.time())
        return wbank

    # test
    def test_only_one_station_returned(self, modified_index_wbank):
        """ensure selecting station 2 only returns one station"""
        station = "2"
        df = modified_index_wbank.read_index(station=station)
        stations = df.station.unique()
        assert len(stations) == 1
        assert set(stations) == {station}


class TestFilesWithDifferentFormats:
    """
    Files saved to index with different formats (other than specified)
    should still be readable.
    """

    start = UTC("2017-09-18")
    end = UTC("2017-09-19")
    seed_ids = ("TA.M17A..VHZ", "TA.BOB..VHZ")
    sampling_rate = 1
    format_key = dict(SAC="TA.SAC..VHZ")

    # fixtures
    @pytest.fixture(scope="class")
    def het_bank_path(self, class_tmp_dir):
        """
        create a directory that has waveforms saved to many file formats in it.
        """
        new_dir = join(class_tmp_dir, "temp1")
        ardir = ArchiveDirectory(
            new_dir, self.start, self.end, self.sampling_rate, seed_ids=self.seed_ids
        )
        ardir.create_directory()
        # add non-mseed files
        for format, seed_id in self.format_key.items():
            st = ardir.create_stream(self.start, self.end, seed_ids=[seed_id])
            path = join(new_dir, format)
            st.write(path + ".mseed", format)
        return obsplus.WaveBank(ardir.path)

    def test_all_files_read_warning_issued(self, het_bank_path):
        """ensure all the files are read in and a warning issued."""
        with pytest.warns(UserWarning) as w:
            bank = obsplus.WaveBank(het_bank_path).update_index()
        df = bank.read_index()
        assert len(w)
        assert set(self.format_key).issubset(df.station.unique())
