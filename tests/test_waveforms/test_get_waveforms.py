"""
tests for get waveforms
"""
import obspy
import pytest


@pytest.fixture(scope="class")
def stream():
    """ create a test waveforms with several channels """
    st1 = obspy.read()
    # set to different station/channel and times
    st2 = obspy.read()
    t1 = obspy.UTCDateTime("2016-01-01")
    for tr in st2:
        tr.stats.network = "LF"
        tr.stats.station = "BOB"
        tr.stats.starttime = t1
    return st2 + st1


class TestGetWaveforms:
    def test_has_attr(self, stream):
        """ ensure the get_waveforms attrs exists """
        assert hasattr(stream, "get_waveforms")

    def test_filter_on_sid(self, stream):
        """ ensure filtering on station id and such """
        t1 = obspy.UTCDateTime("2016-01-01")
        filt = dict(
            network="LF",
            station="BOB",
            location="*",
            channel="*",
            starttime=t1,
            endtime=t1 + 10,
        )
        st = stream.get_waveforms(**filt)
        assert all([tr.stats.network == "LF" for tr in st])
        assert all([tr.stats.station == "BOB" for tr in st])


class TestGetWaveformsBulk:
    def test_has_attr(self, stream):
        """ ensure the get_waveforms attrs exists """
        assert hasattr(stream, "get_waveforms_bulk")

    def test_filter_on_sid(self, stream):
        """ ensure filtering on station id and such """
        t1 = obspy.UTCDateTime("2016-01-01")
        bulk = [("LF", "BOB", "*", "*", t1, t1 + 10)]
        st = stream.get_waveforms_bulk(bulk)
        assert all([tr.stats.network == "LF" for tr in st])
        assert all([tr.stats.station == "BOB" for tr in st])
