"""
Tests for waveform utilities.
"""
import copy
import inspect
from pathlib import Path

import numpy as np
import obspy
import pytest

import obsplus
from obsplus.constants import NSLC
from obsplus.waveforms.utils import (
    trim_event_stream,
    stream2contiguous,
    archive_to_sds,
    merge_traces,
    stream_bulk_split,
)


class TestTrimEventStream:
    """ ensure the trim_event waveforms function works """

    # fixtures
    @pytest.fixture(scope="class")
    def stream_with_short_end(self):
        """ snip off some waveform from the end, return the new waveforms with
        the time the waveform was snipped """
        st = obspy.read()
        t1, t2 = st[0].stats.starttime, st[0].stats.endtime
        new_t2 = t2 - 10
        st[0].trim(endtime=new_t2)
        return st, new_t2

    # tests
    def test_trimmed(self, stream_with_short_end):
        """
        test that the max time on the waveforms is t2
        """
        st, t2 = stream_with_short_end
        st_new = trim_event_stream(st, required_len=None)
        max_endtime = max([tr.stats.endtime.timestamp for tr in st_new])
        assert abs(max_endtime - t2.timestamp) < 0.1

    def test_disjointed_raises(self, disjointed_stream):
        """
        A disjointed waveforms should raise
        """
        with pytest.raises(ValueError) as e:
            trim_event_stream(disjointed_stream)
            assert "the following waveforms is disjointed" in str(e)

    def test_trim_tolerance(self, stream_with_short_end):
        """ ensure a value error is raised when the difference in start or
        end times exceeds the supplied trim tolerance """
        with pytest.raises(ValueError) as e:
            trim_event_stream(stream_with_short_end[0], trim_tolerance=2.0)
        assert "trim tolerance" in str(e.value.args[0])

    def test_fragmented_stream(self, fragmented_stream):
        """ test with streams that are fragmented """
        with pytest.warns(UserWarning) as w:
            st = trim_event_stream(fragmented_stream)
        assert "seconds long" in str(w[0].message)
        stations = {tr.stats.station for tr in st}
        assert "BOB" not in stations


class TestMegeStream:
    """ Tests for obsplus' style for merging streams together. """

    def convert_stream_dtype(self, st, dtype):
        """ Convert datatypes on each trace in the stream. """
        st = st.copy()
        for tr in st:
            tr.data = tr.data.astype(dtype)
            assert tr.data.dtype == dtype
        return st

    def test_identical_streams(self):
        """ ensure passing identical streams performs de-duplication. """
        st = obspy.read()
        st2 = obspy.read() + st + obspy.read()
        st_out = merge_traces(st2)
        assert st_out == st

    def test_adjacent_traces(self):
        """ Traces that are one sample away in time should be merged together. """
        # create stream with traces adjacent in time and merge together
        st1 = obspy.read()
        st2 = obspy.read()
        for tr1, tr2 in zip(st1, st2):
            tr2.stats.starttime = tr1.stats.endtime + 1.0 / tr2.stats.sampling_rate
        st_in = st1 + st2
        out = merge_traces(st_in)
        assert len(out) == 3
        # should be the same as merge and split
        assert out == st_in.merge(1).split()

    def test_traces_with_overlap(self):
        """ Trace with overlap should be merged together. """
        st1 = obspy.read()
        st2 = obspy.read()
        for tr1, tr2 in zip(st1, st2):
            tr2.stats.starttime = tr1.stats.starttime + 10
        st_in = st1 + st2
        out = merge_traces(st_in)
        assert out == st_in.merge(1).split()

    def test_traces_with_different_sampling_rates(self):
        """ traces with different sampling_rates should be left alone. """
        st1 = obspy.read()
        st2 = obspy.read()
        for tr in st2:
            tr.stats.sampling_rate = tr.stats.sampling_rate * 2
        st_in = st1 + st2
        st_out = merge_traces(st_in)
        assert st_out == st_in

    def test_array_data_type(self):
        """ The array datatype should not change. """
        # test floats
        st1 = obspy.read()
        st2 = obspy.read()
        st_out1 = merge_traces(st1 + st2)
        for tr1, tr2 in zip(st_out1, st1):
            assert tr1.data.dtype == tr2.data.dtype
        # tests ints
        st3 = self.convert_stream_dtype(st1, np.int32)
        st4 = self.convert_stream_dtype(st1, np.int32)
        st_out2 = merge_traces(st3 + st4)
        for tr in st_out2:
            assert tr.data.dtype == np.int32
        # def test one int one float
        st_out3 = merge_traces(st1 + st3)
        for tr in st_out3:
            assert tr.data.dtype == np.float64
        # ensure order of traces doesn't mater for dtypes
        st_out4 = merge_traces(st3 + st1)
        for tr in st_out4:
            assert tr.data.dtype == np.float64


class TestStream2Contiguous:
    """ test the stream2contiguous function works """

    # helper functions
    @staticmethod
    def streams_are_equal(st1, st2):
        """ test that the streams are equal minus the processing attr of
        stats dict """
        st1.sort()
        st2.sort()
        for tr1, tr2 in zip(st1.traces, st2.traces):
            if not np.array_equal(tr1.data, tr2.data):
                return False
            d1 = copy.deepcopy(tr1.stats)
            d1.pop("processing", None)
            d2 = copy.deepcopy(tr2.stats)
            d2.pop("processing", None)
            if not d1 == d2:
                return False
        return True

    # fixtures
    @pytest.fixture(scope="class")
    def one_trace_gap_overlaps_stream(self):
        """ a waveforms a gap on one trace """
        st = obspy.read()
        st1 = st.copy()
        st2 = st.copy()
        t1 = st[0].stats.starttime
        t2 = st[0].stats.endtime
        average = obspy.UTCDateTime((t1.timestamp + t2.timestamp) / 2.0)
        a1 = average - 1
        a2 = average + 1
        st1[0].trim(starttime=t1, endtime=a1)
        st2[0].trim(starttime=a2, endtime=t2)
        st = st1 + st2
        return st

    # tests
    def test_contiguous(self, basic_stream_with_gap):
        st, st1, st2 = basic_stream_with_gap
        out = stream2contiguous(st)
        assert inspect.isgenerator(out)
        slist = list(out)
        assert len(slist) == 2
        st_out_1 = slist[0]
        st_out_2 = slist[1]
        # lengths should be equal
        assert len(st_out_1) == len(st1)
        assert len(st_out_2) == len(st2)
        # streams should be equal
        assert self.streams_are_equal(st_out_1, st1)
        assert self.streams_are_equal(st_out_2, st2)

    def test_disjoint(self, disjointed_stream):
        """ ensure nothing is returned if waveforms has not times were all
        three channels have data """
        out = stream2contiguous(disjointed_stream)
        assert inspect.isgenerator(out)
        slist = list(out)
        assert not len(slist)

    def test_one_trace_gap(self, one_trace_gap_overlaps_stream):
        """ ensure nothing is returned if waveforms has not times were all
        three channels have data """
        st = one_trace_gap_overlaps_stream
        out = stream2contiguous(st)
        assert inspect.isgenerator(out)
        slist = list(out)
        assert len(slist) == 2
        for st_out in slist:
            assert not len(st_out.get_gaps())


class TestArchiveToSDS:
    """ Tests for converting archives to SDS. """

    stream_process_count = 0
    dataset_name = "kemmerer"  # dataset used for testing

    def stream_processor(self, st):
        self.stream_process_count += 1
        return st

    @pytest.fixture(scope="class")
    def converted_archive(self, tmpdir_factory):
        """ Convert a dataset archive to a SDS archive. """
        out = tmpdir_factory.mktemp("new_sds")
        ds = obsplus.load_dataset(self.dataset_name)
        wf_path = ds.waveform_path
        archive_to_sds(wf_path, out, stream_processor=self.stream_processor)
        # Because fixtures run in different context then tests this we
        # need to test that the stream processor ran here.
        assert self.stream_process_count
        return out

    @pytest.fixture(scope="class")
    def sds_wavebank(self, converted_archive):
        """ Create a new WaveBank on the converted archive. """
        wb = obsplus.WaveBank(converted_archive)
        wb.update_index()
        return wb

    @pytest.fixture(scope="class")
    def old_wavebank(self):
        """ get the wavebank of the archive before converting to sds """
        ds = obsplus.load_dataset(self.dataset_name)
        bank = ds.waveform_client
        assert isinstance(bank, obsplus.WaveBank)
        return bank

    def test_path_exists(self, converted_archive):
        """ ensure the path to the new SDS exists """
        path = Path(converted_archive)
        assert path.exists()

    def test_directory_not_empty(self, sds_wavebank, old_wavebank):
        """ ensure the same date range is found in the new archive """
        sds_index = sds_wavebank.read_index()
        old_index = old_wavebank.read_index()
        # start times and endtimes for old and new should be the same
        group_old = old_index.groupby(list(NSLC))
        group_sds = sds_index.groupby(list(NSLC))
        # ensure starttimes are the same
        old_start = group_old.starttime.min()
        sds_start = group_sds.starttime.min()
        assert (old_start == sds_start).all()
        # ensure endtimes are the same
        old_end = group_old.endtime.max()
        sds_end = group_sds.endtime.max()
        assert (old_end == sds_end).all()

    def test_each_file_one_trace(self, sds_wavebank):
        """ ensure each file in the sds has exactly one channel """
        index = sds_wavebank.read_index()
        for fi in index.path.unique():
            base = Path(sds_wavebank.bank_path) / fi[1:]
            st = obspy.read(str(base))
            assert len({tr.id for tr in st}) == 1


class TestStreamBulkSplit:
    """ Tests for converting a trace to a list of Streams. """

    @pytest.fixture
    def multi_stream(self):
        """ Create two streams with different station names/channels """
        st1 = obspy.read()
        st2 = obspy.read()
        for tr in st2:
            tr.stats.station = "BOB"
            tr.stats.channel = "HH" + tr.stats.channel[-1]
        return st1 + st2

    def get_bulk_from_stream(self, st, tr_inds, times):
        """ Create a bulk argument from a stream for traces specified and
        relative times. """
        out = []
        for tr_ind, times in zip(tr_inds, times):
            tr = st[tr_ind]
            nslc = tr.id.split(".")
            t1 = tr.stats.starttime + times[0]
            t2 = tr.stats.endtime + times[1]
            out.append(tuple(nslc + [t1, t2]))
        return out

    def test_stream_bulk_split(self):
        """ Ensure the basic stream to trace works. """
        # get bulk params
        st = obspy.read()
        t1, t2 = st[0].stats.starttime + 1, st[0].stats.endtime - 1
        nslc = st[0].id.split(".")
        bulk = [tuple(nslc + [t1, t2])]
        # create traces, check len
        streams = stream_bulk_split(st, bulk)
        assert len(streams) == 1
        # assert trace after trimming is equal to before
        t_expected = obspy.Stream([st[0].trim(starttime=t1, endtime=t2)])
        assert t_expected == streams[0]

    def test_empty_query_returns_empty(self):
        """ An empty query should return an emtpy Stream """
        st = obspy.read()
        out = stream_bulk_split(st, [])
        assert len(out) == 0

    def test_empy_stream_returns_empty(self):
        """ An empy stream should also return an empty stream """
        st = obspy.read()
        t1, t2 = st[0].stats.starttime + 1, st[0].stats.endtime - 1
        nslc = st[0].id.split(".")
        bulk = [tuple(nslc + [t1, t2])]
        out = stream_bulk_split(obspy.Stream(), bulk)
        assert len(out) == 0

    def test_no_bulk_matches(self):
        """ Test when multiple bulk parameters don't match any traces. """
        st = obspy.read()
        bulk = []
        for tr in st:
            utc = obspy.UTCDateTime("2017-09-18")
            t1, t2 = utc, utc
            bulk.append(tuple([*tr.id.split(".") + [t1, t2]]))
        out = stream_bulk_split(st, bulk)
        assert len(out) == len(bulk)
        for tr in out:
            assert isinstance(tr, obspy.Stream)

    def test_two_overlap(self):
        """ Tests for when there is an overlap of available data and
        requested data but some data are not available."""
        # setup stream and bulk args
        st = obspy.read()
        duration = st[0].stats.endtime - st[0].stats.starttime
        bulk = self.get_bulk_from_stream(st, [0, 1], [[-5, -5], [-5, -5]])
        # request data, check durations
        out = stream_bulk_split(st, bulk)
        for st_out in out:
            assert len(st_out) == 1
            stats = st_out[0].stats
            out_duration = stats.endtime - stats.starttime
            assert np.isclose(duration - out_duration, 5)

    def test_two_inter(self):
        """ Tests for getting data completely contained in available range.  """
        # setup stream and bulk args
        st = obspy.read()
        duration = st[0].stats.endtime - st[0].stats.starttime
        bulk = self.get_bulk_from_stream(st, [0, 1], [[5, -5], [5, -5]])
        # request data, check durations
        out = stream_bulk_split(st, bulk)
        for st_out in out:
            assert len(st_out) == 1
            stats = st_out[0].stats
            out_duration = stats.endtime - stats.starttime
            assert np.isclose(duration - out_duration, 10)

    def test_two_intervals_same_stream(self):
        """ Tests for returning two intervals in the same stream. """
        st = obspy.read()
        bulk = self.get_bulk_from_stream(st, [0, 0], [[0, -15], [15, 0]])
        out = stream_bulk_split(st, bulk)
        assert len(out) == 2
        for st_out in out:
            assert len(st_out) == 1
            stats = st_out[0].stats
            out_duration = stats.endtime - stats.starttime
            assert abs(out_duration - 15) <= stats.sampling_rate * 2
