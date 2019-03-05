""" test for core functionality of wavebank """
import functools
import glob
import os
import pathlib
import shutil
import sys
import tempfile
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from os.path import join
from pathlib import Path

import numpy as np
import obsplus
import obsplus.bank.utils
import obsplus.bank.wavebank as sbank
import obspy
import obspy.clients.fdsn
import pandas as pd
import pytest
from obsplus.bank.wavebank import WaveBank
from obsplus.constants import NSLC, NULL_NSLC_CODES
from obsplus.exceptions import BankDoesNotExistError
from obsplus.utils import make_time_chunks, iter_files, get_reference_time
from obspy import UTCDateTime as UTC


# ----------------------------------- Helper functions


def count_calls(instance, bound_method, counter_attr):
    """ given an instance, count how many times bound_method is called and
    store the result on instance in an attr named counter_attr """
    # set counter
    setattr(instance, counter_attr, 0)

    @functools.wraps(bound_method)
    def wraper(*args, **kwargs):
        out = bound_method(*args, **kwargs)
        # increment counter
        setattr(instance, counter_attr, getattr(instance, counter_attr) + 1)
        return out

    return wraper


def strip_processing(st: obspy.Stream) -> obspy.Stream:
    """ strip out processing from stats """
    # TODO replace this when ObsPy #2286 gets merged
    for tr in st:
        tr.stats.pop("processing")
    return st


class ArchiveDirectory:
    """ class for creating a simple archive """

    def __init__(
        self,
        path,
        starttime=None,
        endtime=None,
        sampling_rate=1,
        duration=3600,
        overlap=0,
        gaps=None,
        seed_ids=("TA.M17A..VHZ", "TA.BOB..VHZ"),
    ):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        self.starttime = starttime
        self.endtime = endtime
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.overlap = overlap
        self.seed_ids = seed_ids
        self.gaps = gaps

    def create_stream(self, starttime, endtime, seed_ids=None, sampling_rate=None):
        """ create a waveforms from random data """
        t1 = obspy.UTCDateTime(starttime)
        t2 = obspy.UTCDateTime(endtime)
        sr = sampling_rate or self.sampling_rate
        ar_len = int((t2.timestamp - t1.timestamp) * sr)
        st = obspy.Stream()
        for seed in seed_ids or self.seed_ids:
            n, s, l, c = seed.split(".")
            meta = {
                "sampling_rate": sr,
                "starttime": t1,
                "network": n,
                "station": s,
                "location": l,
                "channel": c,
            }
            data = np.random.randn(ar_len)
            tr = obspy.Trace(data=data, header=meta)
            st.append(tr)
        return st

    def get_gap_stream(self, t1, t2, gaps):
        """ return streams with gaps in it """
        assert len(gaps) == 1
        gap = gaps.iloc[0]
        ts1, ts2 = t1.timestamp, t2.timestamp
        # if gap covers time completely
        if gap.start <= ts1 and gap.end >= ts2:
            raise ValueError("gapped out")
        # if gap is contained by time frame
        elif gap.start > ts1 and gap.end < ts2:
            st1 = self.create_stream(ts1, gap.start)
            st2 = self.create_stream(gap.end, ts2)
            return st1 + st2
        # if gap only effects endtime
        elif ts1 < gap.start < ts2 <= gap.end:
            return self.create_stream(ts1, gap.start)
        # if gap only effects starttime
        elif gap.start <= ts1 < gap.end < ts2:
            return self.create_stream(gap.end, ts2)
        else:  # should not reach here
            raise ValueError("something went very wrong!")

    def create_directory(self):
        """ create the directory with gaps in it """
        # get a dataframe of the gaps
        if self.gaps is not None:
            df = pd.DataFrame(self.gaps, columns=["start", "end"])
            df["start"] = df["start"].apply(lambda x: x.timestamp)
            df["end"] = df["end"].apply(lambda x: x.timestamp)
        else:
            df = pd.DataFrame(columns=["start", "end"])

        assert self.starttime and self.endtime, "needs defined times"
        for t1, t2 in make_time_chunks(
            self.starttime, self.endtime, self.duration, self.overlap
        ):
            # figure out of this time lies in a gap
            gap = df[~((df.start >= t2) | (df.end <= t1))]
            if not gap.empty:
                try:
                    st = self.get_gap_stream(t1, t2, gap)
                except ValueError:
                    continue
            else:
                st = self.create_stream(t1, t2)
            finame = str(t1).split(".")[0].replace(":", "-") + ".mseed"
            path = join(self.path, finame)
            st.write(path, "mseed")

    def create_directory_from_bulk_args(self, bulk_args):
        """ Create a directory from bulk waveform arguments """
        # ensure directory exists
        path = Path(self.path)
        path.mkdir(exist_ok=True, parents=True)
        for (net, sta, loc, chan, t1, t2) in bulk_args:
            nslc = ".".join([net, sta, loc, chan])
            st = self.create_stream(t1, t2, (nslc,))
            time_name = str(t1).split(".")[0].replace(":", "-") + ".mseed"
            save_name = path / f"{net}_{sta}_{time_name}"
            st.write(str(save_name), "mseed")


# ------------------------------ Fixtures


@pytest.fixture(scope="class")
def ta_bank(tmp_ta_dir):
    """ init a bank on the test TA dataset """
    inventory_path = os.path.join(tmp_ta_dir, "inventory.xml")
    bank_path = os.path.join(tmp_ta_dir, "waveforms")
    return sbank.WaveBank(bank_path, inventory=inventory_path)


@pytest.fixture(scope="function")
def ta_bank_index(ta_bank):
    """ return the ta bank, but first update index """
    ta_bank.update_index()
    return ta_bank


@pytest.fixture(scope="function")
def ta_bank_no_index(ta_bank):
    """ return the ta bank, but first delete the index """
    path = Path(ta_bank.index_path)
    if path.exists():
        os.remove(path)
    return ta_bank


@pytest.fixture(scope="function")
def ta_index(ta_bank_index):
    """ return the ta index """
    return ta_bank_index.read_index()


@pytest.fixture
def default_bank(tmpdir):
    """ create a  directory out of the traces in default waveforms, init bank """
    base = Path(tmpdir)
    st = obspy.read()
    for num, tr in enumerate(st):
        name = base / f"{(num)}.mseed"
        tr.write(str(name), "mseed")
    bank = WaveBank(base)
    bank.update_index()
    return bank


# ------------------------------ Tests


class TestBankBasics:
    """ basic tests for Bank class """

    # fixtures
    @pytest.fixture(scope="function")
    def ta_bank_no_index(self, ta_bank):
        """ return the ta bank, but first delete the index if it exists """
        index_path = os.path.join(ta_bank.bank_path, ".index.h5")
        if os.path.exists(index_path):
            os.remove(index_path)
        return sbank.WaveBank(ta_bank.bank_path)

    @pytest.fixture
    def default_bank_low_version(self, default_bank):
        """ return the default bank with a negative version number. """
        # monkey patch obsplus version
        negative_version = "0.0.-1"
        version = obsplus.__version__
        obsplus.__version__ = negative_version
        # write index with negative version
        os.remove(default_bank.index_path)
        default_bank.update_index()
        assert default_bank._index_version == negative_version
        # restore correct version
        obsplus.__version__ = version
        return default_bank

    # tests
    def test_type(self, ta_bank):
        """ make sure TA bank is a bank """
        assert isinstance(ta_bank, sbank.WaveBank)

    def test_index(self, ta_bank_index):
        """ ensure the index exists """
        assert os.path.exists(ta_bank_index.index_path)
        assert isinstance(ta_bank_index.last_updated, float)

    def test_create_index(self, ta_bank_no_index):
        """ make sure a fresh index can be created """
        # test that just trying to get an index that doesnt exists creates it
        ta_bank_no_index.read_index()
        index_path = ta_bank_no_index.index_path
        bank_path = ta_bank_no_index.bank_path
        assert os.path.exists(index_path)
        # make sure all file paths are in the index
        index = ta_bank_no_index.read_index()
        file_paths = set(ta_bank_no_index.bank_path + index.path)
        for file_path in iter_files(bank_path, ext="mseed"):
            # go up two levels to match path reference
            file_path = os.path.abspath(file_path)
            assert file_path in file_paths

    def test_update_index_bumps_only_for_new_files(self, ta_bank_index):
        """ test that updating the index does not modify the last_updated
         time if no new files were added """
        last_updated1 = ta_bank_index.last_updated
        ta_bank_index.update_index()
        last_updated2 = ta_bank_index.last_updated
        # updating should not get stamped unless files were added
        assert last_updated1 == last_updated2

    def test_pathlib_object(self, tmp_ta_dir):
        """ ensure a pathlib object can be passed as first arg """
        bank = WaveBank(pathlib.Path(tmp_ta_dir) / "waveforms")
        ind = bank.read_index()
        min_start = ind.starttime.min()
        st = bank.get_waveforms(starttime=min_start, endtime=min_start + 600)
        assert isinstance(st, obspy.Stream)

    def test_starttime_larger_than_endtime_raises(self, ta_bank):
        """
        If any function that reads index is given a starttime larger
        than the endtime a ValueError should be raised
        """
        with pytest.raises(ValueError) as e:
            ta_bank.read_index(starttime=10, endtime=1)
        assert "starttime cannot be greater than endtime" in str(e)

    def test_correct_endtime_in_index(self, default_bank):
        """ ensure the index has times consistent with traces in waveforms """
        index = default_bank.read_index()
        st = obspy.read()
        starttimes = [tr.stats.starttime.timestamp for tr in st]
        endtimes = [tr.stats.endtime.timestamp for tr in st]

        assert min(starttimes) == index.starttime.min()
        assert max(endtimes) == index.endtime.max()

    def test_bank_can_init_bank(self, default_bank):
        """ WaveBank should be able to take a Wavebank as an input arg. """
        bank = obsplus.WaveBank(default_bank)
        assert isinstance(bank, obsplus.WaveBank)

    def test_stream_is_not_instance(self):
        """ A waveforms object should not be an instance of WaveBank. """
        assert not isinstance(obspy.read(), WaveBank)

    def test_min_version_recreates_index(self, default_bank_low_version):
        """
        If the min version is not met the index should be deleted and re-created.
        A warning should be issued.
        """
        bank = default_bank_low_version
        ipath = Path(bank.index_path)
        mtime1 = ipath.stat().st_mtime
        with pytest.warns(UserWarning) as w:
            bank.update_index()
        assert len(w)  # a warning should have been raised
        mtime2 = ipath.stat().st_mtime
        # ensure the index was deleted and rewritten
        assert mtime1 < mtime2

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
        bank.put_waveforms(obspy.read())
        assert len(bank.read_index()) == 3


class TestEmptyBank:
    """ tests for graceful handling of empty sbanks"""

    @pytest.fixture(scope="class")
    def empty_bank(self):
        """ init a bank with an empty directory, return """
        with tempfile.TemporaryDirectory() as td:
            yield WaveBank(td)

    @pytest.fixture(scope="class")
    def empty_index(self, empty_bank):
        """ return the result of the empty read_index """
        return empty_bank.read_index()

    # tests
    def test_empty_index_returned(self, empty_index):
        """ ensure an empty index (df) was returned """
        assert isinstance(empty_index, pd.DataFrame)
        assert empty_index.empty
        assert set(empty_index.columns).issuperset(WaveBank.index_columns)


class TestGetIndex:
    """ tests for getting the index """

    # tests
    def test_attr(self, ta_bank_index):
        """ test that the get index attribute exists """
        assert hasattr(ta_bank_index, "read_index")

    def test_basic_filters1(self, ta_bank_index):
        """ test exact matches work """
        index = ta_bank_index.read_index(station="M11A")
        assert set(index.station) == {"M11A"}

    def test_basic_filters2(self, ta_bank_index):
        """ that ? work in search """
        index = ta_bank_index.read_index(station="M11?")
        assert set(index.station) == {"M11A"}

    def test_basic_filters3(self, ta_bank_index):
        """ general wildcard searches work """
        index = ta_bank_index.read_index(station="*")
        assert set(index.station) == {"M11A", "M14A"}

    def test_basic_filters4(self, ta_bank_index):
        """ test combinations of wildcards work """
        index = ta_bank_index.read_index(station="M*4?")
        assert set(index.station) == {"M14A"}

    def test_time_queries(self, ta_bank_index):
        """ test that time queries return all data in range """
        t1 = obspy.UTCDateTime("2007-02-15T00-00-00")
        t2 = t1.timestamp + 3600 * 60
        index = ta_bank_index.read_index(starttime=t1, endtime=t2)
        # ensure no data in the index fall far out of bounds of the query time
        # some may be slightly out of bounds due to the buffers used
        assert (index.endtime < (t1 - 3600)).sum() == 0
        assert (index.starttime > (t2 + 3600)).sum() == 0

    def test_time_queries2(self, ta_bank_index):
        """ test that queries that are less than a file size work """
        t1 = obspy.UTCDateTime("2007-02-16T01-01-01")
        t2 = obspy.UTCDateTime("2007-02-16T01-01-15")
        index = ta_bank_index.read_index(starttime=t1, endtime=t2)
        starttime = index.starttime.min()
        endtime = index.endtime.max()
        assert t1.timestamp >= starttime
        assert t2.timestamp <= endtime
        assert not index.empty

    def test_crandall_query(self, crandall_dataset):
        """ tests that querying the crandall dataset's event_waveforms.
        There was one event that didn't correctly get the waveforms
        on read index. """
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
        assert not (df == "None").any().any()


class TestYieldStreams:
    """ tests for yielding streams from the bank """

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
        """ the first yield set of parameters, duration not defined """
        return ta_bank_index.yield_waveforms(**self.query1)

    @pytest.fixture(scope="function")
    def yield2(self, ta_bank_index):
        """ the second yield set of parameters, duration and overlap
         are used """
        return ta_bank_index.yield_waveforms(**self.query2)

    @pytest.fixture
    def yield_file_by_time(self):
        """ Read all files that have data for a sequence of times, yield """

    # tests
    def test_type(self, yield1):
        """ test that calling yield_waveforms returns a generator """
        assert isinstance(yield1, types.GeneratorType)
        count = 0
        for st in yield1:
            assert isinstance(st, obspy.Stream)
            assert len(st)
            count += 1
        assert count  # ensure some files were found

    def test_yield_with_durations(self, yield2, ta_index):
        """ when durations is used each waveforms should have all the
        channels"""
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
    """ tests for getting waveforms from the index """

    # fixture params
    query1 = {
        "starttime": obspy.UTCDateTime("2007-02-15T00-00-10"),
        "endtime": obspy.UTCDateTime("2007-02-20T00-00-00"),
        "station": "*",
        "network": "*",
        "channel": "VHE",
        "attach_response": True,
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
        """ return the waveforms using query1 as params """
        with pytest.warns(UserWarning) as _:
            out = ta_bank.get_waveforms(**self.query1)
        return out

    @pytest.fixture(scope="class")
    def stream2(self, ta_bank):
        """ return the waveforms using query2 as params """
        return ta_bank.get_waveforms(**self.query2)

    @pytest.fixture(scope="class")
    def stream3(self, bingham_dataset):
        """ return a waveforms from query params 3 on bingham dataset """
        bank = bingham_dataset.waveform_client
        return bank.get_waveforms(**self.query3)

    @pytest.fixture
    def bank49(self, tmpdir):
        """ setup a WaveBank to test issue #49. """
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
        """ create a bank that has nullish location codes in its streams. """
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
        """ test that the bank class has the get_waveforms attr """
        assert hasattr(ta_bank_index, "get_waveforms")

    def test_stream1(self, stream1):
        """ make sure stream1 has all the expected features """
        assert isinstance(stream1, obspy.Stream)
        assert len(stream1) == 2
        # get some stats from waveforms
        channels = {tr.stats.channel for tr in stream1}
        starttime = min([tr.stats.starttime for tr in stream1])
        endtime = max([tr.stats.endtime for tr in stream1])
        assert len(channels) == 1
        assert abs(starttime - self.query1["starttime"]) <= 1.0
        assert abs(endtime - self.query1["endtime"]) <= 1.0

    def test_attach_response(self, stream1):
        """ make sure the responses were attached """
        Response = obspy.core.inventory.response.Response
        for tr in stream1:
            assert hasattr(tr.stats, "response")
            assert isinstance(tr.stats.response, Response)

    def test_bracket_matches(self, stream2):
        """ make sure the bracket style filters work (eg VH[NE] for both
        VHN and VHE"""
        channels = {tr.stats.channel for tr in stream2}
        assert channels == {"VHN", "VHE"}

    def test_filter_with_multiple_trace_files(self, crandall_bank):
        """ Ensure a bank with can be correctly filtered. """
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
        """ ensure parameters can be passed as lists """
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


class TestUpdateBar:
    """ tests for the update bar and hook """

    class Bar:
        def __init__(self, max_value=100, min_value=None):
            self.out = dict(count=0)
            self.max = max

        def update(self, current):
            self.out["count"] += 1

        def finish(self):
            pass

        def __call__(self, *args, **kwargs):
            return self

    def test_update_hook(self, ta_bank_no_index):
        """ test custom bar works """
        bar = self.Bar()
        ta_bank_no_index.update_index(bar=bar, min_files_for_bar=0)
        assert bar.out["count"] > 0

    def test_update_bar_default(self, ta_bank_no_index, monkeypatch):
        """ ensure the default progress bar shows up. """
        stringio = StringIO()
        monkeypatch.setattr(sys, "stdout", stringio)
        ta_bank_no_index.update_index(min_files_for_bar=0)
        stringio.seek(0)
        out = stringio.read()
        assert "updating or creating" in out


class TestGetBulkWaveforms:
    """ tests for pulling multiple waveforms using get_bulk_waveforms """

    t1, t2 = obspy.UTCDateTime("2007-02-16"), obspy.UTCDateTime("2007-02-18")
    bulk1 = [("TA", "M11A", "*", "*", t1, t2), ("TA", "M14A", "*", "*", t1, t2)]
    standard_query1 = {"station": "M1?A", "starttime": t1, "endtime": t2}

    # fixtures
    @pytest.fixture(scope="class")
    def ta_bulk_1(self, ta_bank):
        """ perform the first bulk query and return the result """
        return strip_processing(ta_bank.get_waveforms_bulk(self.bulk1))

    @pytest.fixture(scope="class")
    def ta_standard_1(self, ta_bank):
        """ perform the standard query, return the result """
        st = ta_bank.get_waveforms(**self.standard_query1)
        return strip_processing(st)

    @pytest.fixture
    def bank_3(self, tmpdir_factory):
        """ Create a bank with several different types of streams. """
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
        """ the two queries should return the same streams """
        assert ta_bulk_1 == ta_standard_1

    def test_bulk3_no_matches(self, bank_3):
        """ tests for bizarre wildcard usage. Should not return no data. """
        bulk = [
            ("TA", "*", "*", "HHZ", self.t1, self.t2),
            ("*", "MOB", "*", "*", self.t1, self.t2),
            ("BB", "BOB", "1?", "*", self.t1, self.t2),
        ]
        st = bank_3.get_waveforms_bulk(bulk)
        assert len(st) == 0, "no waveforms should have been returned!"

    def test_bulk3_one_match(self, bank_3):
        """ Another bulk request that should return one trace. """
        bulk = [
            ("TA", "*", "12", "???", self.t1, self.t2),
            ("*", "*", "*", "CHZ", self.t1, self.t2),
        ]
        st = bank_3.get_waveforms_bulk(bulk)
        assert len(st) == 1


class TestGetWaveformsBySeedId:
    """ tests for getting waveforms using seed_ids """

    t1, t2 = TestGetBulkWaveforms.t1, TestGetBulkWaveforms.t2
    seed_ids = ["TA.M14A..VHE", "TA.M11A..VHE"]

    # fixtures
    @pytest.fixture(scope="class")
    def query_2_seeds(self, ta_bank):
        """ return the query using seed_ids """
        return ta_bank.get_waveforms_by_seed(self.seed_ids, self.t1, self.t2)

    @pytest.fixture(scope="class")
    def query_1_seed(self, ta_bank):
        """ query using only one seed_id"""

    # tests
    def test_only_desired_channels(self, query_2_seeds):
        """ ensure only 2 channels were returned """
        assert len(query_2_seeds) == 2
        seed_set = {x.id for x in query_2_seeds}
        assert seed_set == set(self.seed_ids)


class TestBankCache:
    """ test that the time cache avoids repetitive queries to the h5 index """

    query_1 = TestGetBulkWaveforms.standard_query1

    # fixtures
    @pytest.fixture(scope="class")
    def mp_ta_bank(self, ta_bank):
        """ monkey patch the ta_bank instance to count how many times the .h5
        index is accessed, store it on the accessed_times in the ._cache.times
        """
        func = count_calls(ta_bank, ta_bank._index_cache._get_index, "index_calls")
        ta_bank._index_cache._get_index = func
        return ta_bank

    @pytest.fixture(scope="class")
    def query_twice_mp_ta(self, mp_ta_bank):
        """ query the instrumented TA bank twice """
        _ = mp_ta_bank.read_index(**self.query_1)
        _ = mp_ta_bank.read_index(**self.query_1)
        return mp_ta_bank

    # tests
    def test_query_twice(self, query_twice_mp_ta):
        """ make sure a double query only accesses index once """
        assert query_twice_mp_ta.index_calls == 1


class TestBankCacheWithKwargs:
    """ kwargs should get hashed as well, so the same times with different
    kwargs should be cached """

    @pytest.fixture
    def got_gaps_bank(self, ta_bank):
        """ call get_gaps on bank (to populate cache) and return """
        ta_bank.get_gaps_df()
        return ta_bank

    def test_get_gaps_doesnt_overwrite_cache(self, got_gaps_bank):
        """ ensure that calling get gaps doesn't result in read_index
        not returning the path column """
        inds = got_gaps_bank.read_index()
        assert "path" in inds.columns


class TestPutWaveForm:
    """ test that waveforms can be put into the bank """

    # fixtures
    @pytest.fixture(scope="class")
    def add_stream(self, ta_bank):
        """ add the default obspy waveforms to the bank, return the bank """
        st = obspy.read()
        ta_bank.update_index()  # make sure index cache is set
        ta_bank.put_waveforms(st)
        ta_bank.update_index()
        return ta_bank

    @pytest.fixture(scope="class")
    def default_stations(self):
        """ return the default stations on the default waveforms as a set """
        return set([x.stats.station for x in obspy.read()])

    # tests
    def test_deposited_waveform(self, add_stream, default_stations):
        """ make sure the waveform was added to the bank """
        assert default_stations.issubset(add_stream.read_index().station)

    def test_retrieve_stream(self, add_stream):
        """ ensure the default waveforms can be pulled out of the archive """
        st1 = add_stream.get_waveforms(station="RJOB").sort()
        st2 = obspy.read().sort()
        assert len(st1) == 3
        for tr1, tr2 in zip(st1, st2):
            assert np.all(tr1.data == tr2.data)

    def test_put_waveforms_to_crandall_copy(self, tmpdir):
        """ ran into issue in docs where putting data into the crandall
        copy didn't work. """
        ds = obsplus.copy_dataset(dataset="crandall", destination=Path(tmpdir))
        bank = WaveBank(ds.waveform_client)
        bank.read_index()  # this sets cache
        st = obspy.read()
        bank.put_waveforms(st)
        bank.update_index()
        df = bank.read_index(station="RJOB")
        assert len(df) == len(st)
        assert set(df.station) == {"RJOB"}


class TestPutMultipleTracesOneFile:
    """ ensure that multiple waveforms can be put into one file """

    st = obspy.read()
    st_mod = st.copy()
    for tr in st_mod:
        tr.stats.station = "PS"
    expected_seeds = {tr.id for tr in st + st_mod}

    # fixtures
    @pytest.fixture(scope="class")
    def bank(self):
        """ return an empty bank for depositing waveforms """
        bd = dict(path_structure="streams/network", name_structure="time")
        with tempfile.TemporaryDirectory() as tempdir:
            out = os.path.join(tempdir, "temp")
            yield WaveBank(out, **bd)

    @pytest.fixture(scope="class")
    def deposited_bank(self, bank: obsplus.WaveBank):
        """ deposit the waveforms in the bank, return the bank """
        bank.put_waveforms(self.st_mod)
        bank.put_waveforms(self.st)
        return bank

    @pytest.fixture(scope="class")
    def mseed_files(self, deposited_bank):
        """ count the number of files """
        bfile = deposited_bank.bank_path
        glo = glob.glob(os.path.join(bfile, "**", "*.mseed"), recursive=True)
        return glo

    @pytest.fixture(scope="class")
    def banked_stream(self, mseed_files):
        st = obspy.Stream()
        for ftr in mseed_files:
            st += obspy.read(ftr)
        return st

    @pytest.fixture(scope="class")
    def number_of_files(self, mseed_files):
        """ count the number of files """
        return len(mseed_files)

    # tests
    def test_one_file(self, number_of_files):
        """ ensure only one file was written """
        assert number_of_files == 1

    def test_all_streams(self, banked_stream):
        """ ensure all the channels in the waveforms where written to
        the bank """
        banked_seed_ids = {tr.id for tr in banked_stream}
        assert banked_seed_ids == self.expected_seeds


class TestBadWaveforms:
    """ test how wavebank handles bad waveforms """

    # fixtures
    @pytest.fixture(scope="class")
    def ta_bank_bad_file(self, ta_bank):
        """ add an unreadable file to the wave bank, then return new bank """
        path = ta_bank.bank_path
        new_file_path = os.path.join(path, "bad_file.mseed")
        with open(new_file_path, "w") as fi:
            fi.write("this is not an mseed file, duh")
        # remove old index if it exists
        if os.path.exists(ta_bank.index_path):
            os.remove(ta_bank.index_path)
        return sbank.WaveBank(path)

    @pytest.fixture
    def ta_bank_empty_files(self, ta_bank, tmpdir):
        """ add many empty files to bank, ensure index still reads all files """
        old_path = Path(ta_bank.bank_path)
        new_path = Path(tmpdir) / "waveforms"
        shutil.copytree(old_path, new_path)
        # create 100 empty files
        for a in range(100):
            new_file_path = new_path / f"{a}.mseed"
            with new_file_path.open("wb") as fi:
                pass
        # remove old index if it exists
        index_path = new_path / (Path(ta_bank.index_path).name)
        if index_path.exists():
            os.remove(index_path)
        bank = sbank.WaveBank(old_path)
        bank.update_index()
        return bank

    # tests
    def test_bad_file_emits_warning(self, ta_bank_bad_file):
        """ ensure an unreadable waveform file will emmit a warning """

        with pytest.warns(UserWarning) as record:
            ta_bank_bad_file.update_index()
        assert len(record)
        expected_str = "obspy failed to read"
        assert any([expected_str in r.message.args[0] for r in record])

    def test_read_index(self, ta_bank_empty_files, ta_bank):
        """ tests for bank with many empty files """
        df1 = ta_bank_empty_files.read_index()
        df2 = ta_bank.read_index()
        assert (df1 == df2).all().all()

    def test_get_non_existent_waveform(self, ta_bank):
        """ Ensure asking for a non-existent station returns empty waveforms. """
        st = ta_bank.get_waveforms(station="RJOB")
        assert isinstance(st, obspy.Stream)
        assert len(st) == 0


class TestFilesWithMultipleChannels:
    """ make sure banks that have multi-channel files (eg events) behave """

    counter = 0

    # fixtures
    @pytest.fixture(autouse=True, scope="class")
    def count_read_executions(self, bank):
        """ patch read to make sure it is called correct number of times """

        def count_decorator(func):
            def wrapper(*args, **kwargs):
                self.counter += 1
                return func(*args, **kwargs)

            return wrapper

        # update index first so we only count event reads
        bank.update_index()

        old_func = obsplus.utils.READ_DICT["mseed"]
        new_func = count_decorator(old_func)
        obsplus.utils.READ_DICT["mseed"] = new_func
        yield
        self.counter = 0
        obsplus.utils.READ_DICT["mseed"] = old_func

    @pytest.fixture(scope="class")
    def multichannel_bank(self):
        """ return a directory with a mseed that has multiple channels """
        st = obspy.read()
        with tempfile.TemporaryDirectory() as tdir:
            path = join(tdir, "test.mseed")
            st.write(path, "mseed")
            yield tdir
        if os.path.exists(tdir):
            shutil.rmtree(tdir)

    @pytest.fixture(scope="class")
    def bank(self, multichannel_bank):
        """ return a wavefetcher using multichannel bank """
        return WaveBank(multichannel_bank)

    @pytest.fixture(scope="class")
    def bulk_args(self):
        """return bulk args for default waveforms """
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
        """ return the result of getting bulk args """
        return bank.get_waveforms_bulk(bulk_args)

    @pytest.fixture(scope="class")
    def number_of_calls(self, bulk_st):
        """ return the nubmer of calls made to read"""
        return self.counter

    # tests
    def test_stream_len(self, bulk_st):
        """ ensure exactly 3 channels are in waveforms """
        assert len(bulk_st) == 3

    def test_read_stream_called_once(self, number_of_calls):
        """ assert the read function was called exactly once """
        assert number_of_calls == 1


class TestGetAvailability:
    """ test that WaveBank will return an availability dataframe """

    # fixtures
    @pytest.fixture(scope="class")
    def avail_df(self, ta_bank):
        """ return the availability dataframe """
        return ta_bank.get_availability_df()

    @pytest.fixture(scope="class")
    def avail(self, ta_bank):
        """ return availability """
        return ta_bank.availability()

    # test
    def test_availability_df(self, avail_df):
        """ test the availability property returns a dataframe """
        assert isinstance(avail_df, pd.DataFrame)
        assert not avail_df.empty

    def test_avail_df_filter(self, ta_bank):
        """ ensure specifying a network/station argument filters df """
        df = ta_bank.get_availability_df(station="M14*", channel="*Z")
        assert len(df) == 1
        assert df.iloc[0].station == "M14A"
        assert df.iloc[0].channel == "VHZ"

    def test_availability(self, avail):
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
    """ test that the get_gaps method returns info about gaps """

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

    overlap = 0

    # fixtures
    @pytest.fixture(scope="class")
    def gappy_dir(self, class_tmp_dir):
        """ create a directory that has gaps in it """
        new_dir = join(class_tmp_dir, "temp1")
        ardir = ArchiveDirectory(
            new_dir,
            self.start,
            self.end,
            self.sampling_rate,
            gaps=self.gaps,
            overlap=self.overlap,
        )
        ardir.create_directory()
        return class_tmp_dir

    @pytest.fixture(scope="class")
    def gappy_bank(self, gappy_dir):
        """ init a sbank on the gappy data """
        bank = WaveBank(gappy_dir)
        # make sure index is updated after gaps are introduced
        if os.path.exists(bank.index_path):
            os.remove(bank.index_path)
        bank._index_cache = obsplus.bank.utils._IndexCache(bank, 5)
        bank.update_index()
        return bank

    @pytest.fixture(scope="class")
    def empty_bank(self):
        """ create a Sbank object initated on an empty directory """
        with tempfile.TemporaryDirectory() as td:
            bank = WaveBank(td)
            yield bank

    @pytest.fixture(scope="class")
    def gap_df(self, gappy_bank):
        """ return a gap df from the gappy bank"""
        return gappy_bank.get_gaps_df()

    @pytest.fixture(scope="class")
    def uptime_df(self, gappy_bank):
        """ return the uptime dataframe from the gappy bank """
        return gappy_bank.get_uptime_df()

    # tests
    def test_gaps_length(self, gap_df):
        """ ensure each of the gaps shows up in df """
        assert isinstance(gap_df, pd.DataFrame)
        assert not gap_df.empty
        group = gap_df.groupby(["network", "station", "location", "channel"])
        for gnum, df in group:
            assert len(df) == len(self.gaps)
            dif = abs(df.gap_duration - self.durations)
            assert (dif < (1.5 * self.sampling_rate)).all()

    def test_uptime_df(self, uptime_df):
        """ ensure the uptime df is of correct type and accurate """
        assert isinstance(uptime_df, pd.DataFrame)
        gap_duration = sum([x[1] - x[0] for x in self.gaps])
        duration = self.end - self.start
        uptime_percent = (duration - gap_duration) / duration
        assert (abs(uptime_df["availability"] - uptime_percent) < 0.001).all()

    def test_empty_directory(self, empty_bank):
        """ ensure an empty bank get_gaps returns and empty df with expected
        columns """
        gaps = empty_bank.get_gaps_df()
        assert not len(gaps)
        assert set(WaveBank.gap_columns).issubset(set(gaps.columns))

    def test_kemmerer_uptime(self, kem_fetcher):
        """ ensure the kemmerer bank returns an uptime df"""
        bank = kem_fetcher.waveform_client
        df = bank.get_uptime_df()
        assert (df["uptime"] == df["duration"]).all()


class TestBadInputs:
    """ ensure wavebank handles bad inputs correctly """

    # tests
    def test_bad_inventory(self, tmp_ta_dir):
        """ ensure giving a bad stations str raises """
        with pytest.raises(Exception):
            sbank.WaveBank(tmp_ta_dir, inventory="some none existent file")


class TestThreadSafeUpdateIndex:
    """ tests to make sure running update index in different threads
    doesn't cause hdf5 to fail """

    worker_count = 3

    # fixtures
    @pytest.fixture(scope="class")
    def thread_pool(self):
        """ return a thread pool """
        with ThreadPoolExecutor(self.worker_count) as executor:
            yield executor

    @pytest.fixture
    def thread_update_index(self, ta_bank, thread_pool):
        """ run a bunch of update index operations in different threads,
        return list of results """
        out = []
        for _ in range(self.worker_count):
            out.append(thread_pool.submit(ta_bank.update_index))
        return list(as_completed(out))

    # tests
    def test_index_update(self, thread_update_index, ta_bank):
        """ ensure the index updated and the threads didn't kill each
        other """
        # get a list of exceptions that occurred
        excs = [x.exception() for x in thread_update_index if x.exception() is not None]
        assert len(excs) == 0


class TestSelectDoesntReturnSuperset:
    """ make sure selecting on an attribute doesnt return a superset of that
    attribute. EG, selecting station '2' should not also return station '22'
    """

    # fixtures
    @pytest.fixture(scope="class")
    def df_index(self):
        """ return a dataframe index for testing """
        stations = [str(x) for x in [2, 2, 2, 22, 22, 22, 222, 222, 222]]
        df = pd.DataFrame(index=range(9), columns=WaveBank.index_columns)
        df["station"] = stations
        df["network"] = "1"
        df["starttime"] = obspy.UTCDateTime().timestamp
        df["endtime"] = df["starttime"] + 3600
        return df

    @pytest.fixture(scope="class")
    def bank(self, df_index, ta_archive):
        """ return a bank with monkeypatched index """
        sbank = WaveBank(ta_archive)
        sbank.update_index = lambda: None
        sbank._index_cache = lambda *args, **kwargs: df_index
        return sbank

    # test
    def test_only_one_station_returned(self, bank):
        """ ensure selecting station 2 only returns one station """
        station = "2"
        df = bank.read_index(station=station)
        stations = df.station.unique()
        assert len(stations) == 1
        assert set(stations) == {station}


class TestFilesWithDifferentFormats:
    """ Files saved to index with different formats (other than specified)
    should still be readable. """

    start = UTC("2017-09-18")
    end = UTC("2017-09-19")
    seed_ids = ("TA.M17A..VHZ", "TA.BOB..VHZ")
    sampling_rate = 1
    format_key = dict(SAC="TA.SAC..VHZ")

    # fixtures
    @pytest.fixture(scope="class")
    def het_bank(self, class_tmp_dir):
        """ create a directory that has multiple file types in it. """
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

    def test_all_files_read_warning_issued(self, het_bank):
        """ ensure all the files are read in and a warning issued. """
        with pytest.warns(UserWarning) as w:
            df = het_bank.read_index()
        assert len(w)
        assert set(self.format_key).issubset(df.station.unique())
