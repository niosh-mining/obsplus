import copy
import inspect

import numpy as np
import obspy
import pytest
from obsplus.waveforms.utils import trim_event_stream, stream2contiguous


class TestTrimEventStream:
    """ ensure the trim_event waveforms function works """

    # fixtures
    @pytest.fixture(scope="class")
    def stream_with_short_end(self):
        """ snip off some waveform from the end, return the new waveforms with
        the time the waveform was snipped """
        st = pytest.waveforms["default"]
        t1, t2 = st[0].stats.starttime, st[0].stats.endtime
        new_t2 = t2 - 10
        st[0].trim(endtime=new_t2)
        return st, new_t2

    # tests
    def test_trimed(self, stream_with_short_end):
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
        assert "trim tolerance" in str(e)

    def test_fragmented_stream(self, fragmented_stream):
        """ test with streams that are fragmented """
        with pytest.warns(UserWarning) as w:
            st = trim_event_stream(fragmented_stream)
        assert "seconds long" in str(w[0].message)
        stations = {tr.stats.station for tr in st}
        assert "BOB" not in stations


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
        st = pytest.waveforms["default"]
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
