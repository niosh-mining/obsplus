"""
Tests for the WaveFrame class.
"""
import obspy
import pandas as pd
import pytest
from obspy import UTCDateTime

from obsplus import WaveFrame
from obsplus.constants import NSLC
from obsplus.exceptions import DataFrameContentError


@pytest.fixture
def st_no_response():
    """ Get a copy of the default trace, remove response. """
    st = obspy.read()
    # drop response for easier stats dtypes
    for tr in st:
        tr.stats.pop("response", None)
    return st


@pytest.fixture
def waveframe(st_no_response) -> WaveFrame:
    """ Create a basic WaveFrame from default stream. """
    return WaveFrame.from_stream(st_no_response)


@pytest.fixture
def waveframe_gap(st_no_response) -> WaveFrame:
    """
    Create a waveframe with a 1 second gap in the middle.
    Also shorten last trace.
    """
    st1, st2 = st_no_response.copy(), st_no_response.copy()
    for tr in st2:
        tr.stats.starttime = st1[0].stats.endtime + 1
    st2[-1].data = st2[-1].data[:-20]
    wf = WaveFrame.from_stream(st1 + st2)
    assert isinstance(wf, WaveFrame)
    assert len(wf) == 6
    return wf


class Testbasics:
    """ Basic tests of waveframe. """


class TestConstructorStats:
    """ Basic tests input to waveframe. """

    t1 = UTCDateTime("2009-08-24T00-20-03")

    def _make_stats_df(self, args):
        names = list(NSLC) + ["starttime", "endtime"]
        df = pd.DataFrame([args], columns=names)
        return df

    def test_wildcard_raises(self):
        """ WaveFrame does not support using wildcards in str params of bulk. """
        bad_nslc = ["*", "bob", "01", "BHZ", self.t1, self.t1 + 10]
        df = self._make_stats_df(bad_nslc)
        # since there is a wildcard it should raise a ValueError
        with pytest.raises(DataFrameContentError):
            WaveFrame(waveforms=obspy.read(), stats=df)

    def test_test_bad_starttime_endtime_raises(self):
        """ Ensure bad starttime/endtimes raise. """
        bad_args = ["UU", "BOB", "01", "BHZ", self.t1, self.t1 - 10]
        df = self._make_stats_df(bad_args)
        # since starttime is after endtime this should raise
        with pytest.raises(DataFrameContentError):
            WaveFrame(waveforms=obspy.read(), stats=df)

    def test_null_values_raise(self):
        """ Null values in any column should raise. """
        bad_args = ["UU", "BOB", "01", None, self.t1, self.t1 - 10]
        df = self._make_stats_df(bad_args)
        with pytest.raises(DataFrameContentError):
            WaveFrame(waveforms=obspy.read(), stats=df)

    def test_date_columns_renamed(self):
        """ Ensure enddate and startdate get renamed to starttime and endtime """
        bulk = ["BW", "RJOB", "", "EHZ", self.t1, self.t1 + 10]
        names = list(NSLC) + ["startdate", "enddate"]
        df = pd.DataFrame([bulk], columns=names)
        st = obspy.read()
        wf = WaveFrame(waveforms=st, stats=df)
        assert {"starttime", "endtime"}.issubset(set(wf.stats.columns))

    def test_gappy_traces(self, waveframe_gap):
        """ Ensure gappy data still works. """

        # there should also be some NaN on the last row
        data = waveframe_gap.data
        null_count = data.isnull().sum(axis=1)
        # all but the last row should have no nulls
        assert (null_count.loc[:4] == 0).all()
        assert null_count.iloc[-1] == 20

    def test_cant_get_waveform_data(self, st_no_response):
        """
        Test that data has a row of NaN for any stats that couldn't get
        waveforms.
        """
        t1 = self.t1 - 100_000
        bulk = ["BW", "RJOB", "", "EHZ", t1, t1 + 10]
        df = self._make_stats_df(bulk)
        wf = WaveFrame(waveforms=st_no_response, stats=df)
        data = wf.data
        assert len(data) == 1
        assert data.isnull().all().all()

    def test_init_waveframe_from_waveframe(self, waveframe):
        """ A waveframe should be valid input to waveframe constructor"""
        wf1 = waveframe
        wf2 = WaveFrame(wf1)
        assert wf1 is not wf2
        assert wf1._df is not wf2._df
        assert (wf1._df == wf2._df).all().all()

    def test_init_waveframe_from_waveframe_df(self, waveframe):
        """ A waveframe can be inited from a dataframe from a waveframe. """
        wf1 = waveframe
        wf2 = WaveFrame(wf1._df)
        assert wf1 is not wf2
        assert wf1._df is not wf2._df
        assert (wf1._df == wf2._df).all().all()

    def test_init_waveframe_from_waveframe_parts(self, waveframe):
        """ A wavefrom should be init'able from a waveframes parts """
        wf1 = waveframe
        wf2 = WaveFrame(waveforms=wf1.data, stats=wf1.stats)
        assert wf1 is not wf2
        assert wf1._df is not wf2._df
        assert (wf1._df == wf2._df).all().all()


class TestToFromStream:
    """ Tests for converting a stream to a WaveFrame. """

    def test_type(self, waveframe):
        assert isinstance(waveframe, WaveFrame)

    def test_to_stream(self, waveframe, st_no_response):
        st = waveframe.to_stream()
        assert isinstance(st, obspy.Stream)
        assert st == st_no_response
