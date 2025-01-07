"""
Tests for ObsPlus' test utilities.
"""

import numpy as np
import obspy
import pytest
from obsplus.utils.testing import assert_streams_almost_equal


class TestAssertStreamsAlmostEqual:
    """Tests for asserting waveforms are equal."""

    @pytest.fixture
    def streams(self):
        """Return two default streams."""
        return obspy.read(), obspy.read()

    def test_unequal_len(self, streams):
        """Traces are not equal if the number of traces is not equal."""
        st1, st2 = streams
        st2.traces = st2.traces[:-1]
        with pytest.raises(AssertionError):
            assert_streams_almost_equal(st1, st2)

    def test_processing_different(self, streams):
        """If processing of stats is different streams should be equal."""
        st1, st2 = streams
        st1.detrend("linear").detrend("linear")
        st2.detrend("linear")
        # This should not raise
        assert_streams_almost_equal(st1, st2)
        # but this should
        with pytest.raises(AssertionError):
            assert_streams_almost_equal(st1, st2, basic_stats=False)

    def test_basic_stats_different(self, streams):
        """Ensure when basic stats are different streams are not almost equal."""
        st1, st2 = streams
        for tr in st1:
            tr.stats.station = "bob"
        with pytest.raises(AssertionError):
            assert_streams_almost_equal(st1, st2)
        with pytest.raises(AssertionError):
            assert_streams_almost_equal(st1, st2, basic_stats=False)

    def test_off_arrays(self, streams):
        """If the arrays are slightly perturbed they should still be equal."""
        st1, st2 = streams
        rnd = np.random.RandomState(0)
        for tr in st1:
            norm = np.max(abs(tr.data) * 100)
            tr.data += rnd.random(len(tr.data)) / norm

    def test_off_by_one(self):
        """Tests for allowing off by one errors"""
        st1 = obspy.read()
        # shorten each stream by 1
        st2 = obspy.read()
        for tr in st2:
            tr.data = tr.data[:-1]
        # off by one should make this not raise
        assert_streams_almost_equal(st1, st2, allow_off_by_one=True)
        # but without it it should raise
        with pytest.raises(AssertionError):
            assert_streams_almost_equal(st1, st2, allow_off_by_one=False)

    def test_off_by_one_case1(self, bingham_dataset, bingham_stream):
        """Coincidental off by one test case"""
        bank = bingham_dataset.waveform_client
        # get query parameters (these were found by accident)
        t1 = obspy.UTCDateTime(2013, 4, 11, 4, 58, 50, 259000)
        t2 = obspy.UTCDateTime(2013, 4, 11, 4, 58, 58, 281000)
        params = ["UU", "NOQ", "01", "HHN", t1, t2]
        # get waveforms from the wavebank and the streams
        st1 = bank.get_waveforms(*params)
        st2 = bingham_stream.get_waveforms(*params)
        # this should not raise
        assert_streams_almost_equal(st1, st2, allow_off_by_one=True)
