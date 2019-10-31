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
def waveframe():
    """ Create a basic WaveFrame from default stream. """
    st = obspy.read()
    # drop response for easier stats dtypes
    for tr in st:
        tr.stats.pop("response", None)
    return WaveFrame.from_stream(st)


@pytest.fixture
def waveframe_gap():
    """ Create a waveframe with a 1 second gap in the middle. """
    st1, st2 = obspy.read(), obspy.read()
    for tr in st2:
        tr.stats.starttime = st1[0].stats.endtime + 1
    return WaveFrame.from_stream(st1 + st2)


class TestInputStats:
    """ Basic tests input Stats dataframe. """

    t1 = UTCDateTime("2017-09-18")

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
        bulk = ["UU", "BOB", "01", "BHZ", self.t1, self.t1 + 10]
        names = list(NSLC) + ["startdate", "enddate"]
        df = pd.DataFrame([bulk], columns=names)
        wf = WaveFrame(waveforms=obspy.read(), stats=df)
        assert {"starttime", "endtime"}.issubset(set(wf.stats.columns))


class TestToFromStream:
    """ Tests for converting a stream to a WaveFrame. """

    def test_type(self, waveframe):
        assert isinstance(waveframe, WaveFrame)

    def test_to_stream(self, waveframe):
        st = waveframe.to_stream()
        assert st == obspy.read()
