"""
Tests for waveform stuff.
"""

import obspy

from obsplus.utils.testing import assert_streams_almost_equal


class TestGetWaveforms:
    """ Tests for get waveforms of stream. """

    def test_no_params(self):
        """ No params should return a copy of stream. """
        st = obspy.read()
        assert_streams_almost_equal(st, st.get_waveforms())

    def test_out_of_bounds_nslc(self):
        """ Requests that contain no data should return empty stream. """
        st = obspy.read()
        out = st.get_waveforms(station="BOB")
        assert isinstance(out, obspy.Stream)
        assert len(out) == 0

    def test_out_of_bounds_time(self):
        """ Requests that contain no data should return empty stream. """
        st = obspy.read()
        time = obspy.UTCDateTime("1975-01-01")
        out = st.get_waveforms(starttime=time, endtime=time + 60000)
        assert isinstance(out, obspy.Stream)
        assert len(out) == 0
