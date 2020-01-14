"""
tests for get waveforms
"""
import copy

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

    def test_none_times(self, stream):
        """ Ensure starttime/endtime can be None. """
        stream.get_waveforms("*", "*", "*", "*", None, None)


class TestGetWaveformsBulk:
    @pytest.fixture(scope="class")
    def bingham_st(self, bingham_dataset):
        """ Return a stream with all data from bingham. """
        return bingham_dataset.waveform_client.get_waveforms()

    @pytest.fixture(scope="class")
    def bingham_bulk_args(self, bingham_st):
        """ Return bulk arguments which encompass all of the bingham dataset """
        bulk = []
        for tr in bingham_st:
            bulk.append(tr.id.split(".") + [tr.stats.starttime, tr.stats.endtime])
        return bulk

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

    def test_empty_bulk(self, stream):
        """ Ensure an empty Stream is returned when bulk is None """
        st = stream.get_waveforms_bulk([])
        assert isinstance(st, obspy.Stream)
        assert len(st) == 0

    def test_doesnt_modify_original(self, bingham_st, bingham_bulk_args):
        """ Ensure the method doesn't modify the original stream or bulk args """
        st1 = copy.deepcopy(bingham_st)
        bulk1 = copy.deepcopy(bingham_bulk_args)
        _ = st1.get_waveforms_bulk(bulk1)
        assert st1 == bingham_st
        assert bulk1 == bingham_bulk_args

    def test_waveform_bulk(self, bingham_st, bingham_bulk_args):
        """ Test that waveform bulk works on Bingham st """
        # make a long bulk arg
        st = bingham_st.get_waveforms_bulk(bingham_bulk_args)
        assert len(st) == len(bingham_st)

    def test_no_matches(self):
        """ Test waveform bulk when no params meet req. """
        t1 = obspy.UTCDateTime("2012-01-01")
        t2 = t1 + 12
        bulk = [("bob", "is", "no", "sta", t1, t2)]
        st = obspy.read()
        stt = st.get_waveforms_bulk(bulk)
        assert isinstance(stt, obspy.Stream)
        assert len(stt) == 0

    def test_one_match(self):
        """ Test waveform bulk when there is one req. that matches """
        t1 = obspy.UTCDateTime("2012-01-01")
        t2 = t1 + 12
        st = obspy.read()
        bulk = [tuple(st[0].id.split(".") + [t1, t2])]
        stt = st.get_waveforms_bulk(bulk)
        assert isinstance(stt, obspy.Stream)
