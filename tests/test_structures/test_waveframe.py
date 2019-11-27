"""
Tests for the WaveFrame class.
"""
import operator
from typing import Union

import numpy as np
import obspy
import pandas as pd
import pytest
from obspy import UTCDateTime

from obsplus import WaveFrame
from obsplus.constants import NSLC
from obsplus.exceptions import DataFrameContentError
from obsplus.utils.testing import handle_warnings


def _make_st_no_response():
    """ Get a copy of the default trace, remove response. """
    st = obspy.read()
    # drop response for easier stats dtypes
    for tr in st:
        tr.stats.pop("response", None)
    return st


st_no_resp = _make_st_no_response()
wf = WaveFrame.from_stream(st_no_resp)


@pytest.fixture
def st_no_response():
    """ Get a copy of the default trace, remove response. """
    return st_no_resp.copy()


@pytest.fixture
def waveframe_from_stream(st_no_response) -> WaveFrame:
    """ Create a basic WaveFrame from default stream. """
    return wf.copy()


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

    def test_get_item(self, waveframe_from_stream):
        """ get item should return the series from the stats df. """
        wf = waveframe_from_stream
        ser = wf["station"]
        assert (ser == "RJOB").all()

    def test_set_item(self, waveframe_from_stream):
        """ Set item should set a column in the stats dataframe. """
        wf = waveframe_from_stream
        wf["station"] = "JOB"
        assert (wf["station"] == "JOB").all()

    def test_add_new_column(self, waveframe_from_stream):
        """ Add a new column to stats. """
        wf = waveframe_from_stream
        wf["new_col"] = "heyo"
        out = wf["new_col"]
        assert isinstance(out, pd.Series)
        assert len(out) == len(wf)

    def test_filtering(self, waveframe_from_stream):
        """ Ensure passing a boolean filters the dataframe. """
        wf = waveframe_from_stream
        ser = wf["channel"].str.endswith("Z")
        wf2 = wf[ser]
        assert len(wf2) == 1
        assert wf2["channel"].str.endswith("Z").all()


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

    def test_init_waveframe_from_waveframe(self, waveframe_from_stream):
        """ A waveframe should be valid input to waveframe constructor"""
        wf1 = waveframe_from_stream
        wf2 = WaveFrame(wf1)
        assert wf1 is not wf2
        assert wf1._df is not wf2._df
        assert wf1 == wf2

    def test_init_waveframe_from_waveframe_df(self, waveframe_from_stream):
        """ A waveframe can be inited from a dataframe from a waveframe. """
        wf1 = waveframe_from_stream
        wf2 = WaveFrame(wf1._df)
        assert wf1 is not wf2
        assert wf1._df is not wf2._df
        assert wf1 == wf2

    def test_init_waveframe_from_waveframe_parts(self, waveframe_from_stream):
        """ A wavefrom should be init'able from a waveframes parts """
        wf1 = waveframe_from_stream
        wf2 = WaveFrame(waveforms=wf1.data, stats=wf1.stats)
        assert wf1 is not wf2
        assert wf1._df is not wf2._df
        assert wf1 == wf2

    def test_waveframe_has_delta(self, waveframe_from_stream):
        """ Waveframe should have a delta parameter in its stats. """
        stats = waveframe_from_stream.stats
        assert "delta" in stats.columns
        assert np.issubdtype(stats["delta"].values.dtype, np.timedelta64)

    def test_stream_uneven(self, st_no_response):
        """ Tests for when streams are not evenly sized. """
        st = st_no_response
        st[0].data = st[0].data[:100]
        wf = WaveFrame.from_stream(st)
        # only 100 values should be Non-null
        assert (~wf.data.loc[0].isnull()).sum() == 100
        # the shape should still be greater than 100
        assert wf.data.shape[-1] > 100


class TestBasicOperations:
    """ Tests for basic operations. """

    numbers = (0, 1, -1, 1.11, 10e2, 1 + 10j)

    def _generic_op_test(self, wf1, number, op):
        """ Apply generic operation tests. """
        # div 0 raises runtime warning, this is ok.
        with handle_warnings():
            wf2 = op(wf1, number)
            data1 = op(wf1.data.values, number)

        assert isinstance(wf2, WaveFrame)
        assert wf2 is not wf1
        data2 = wf2.data.values
        close = np.isclose(data1, data2)
        not_finite = ~np.isfinite(data1)
        assert np.all(close | not_finite)

    @pytest.mark.parametrize("number", numbers)
    def test_add_numbers(self, waveframe_from_stream, number):
        """ Tests for adding numbers. """
        self._generic_op_test(waveframe_from_stream, number, operator.add)

    @pytest.mark.parametrize("number", numbers)
    def test_subtract_numbers(self, waveframe_from_stream, number):
        """ Tests for adding numbers. """
        self._generic_op_test(waveframe_from_stream, number, operator.sub)

    @pytest.mark.parametrize("number", numbers)
    def test_mult_numbers(self, waveframe_from_stream, number):
        """ Tests for adding numbers. """
        self._generic_op_test(waveframe_from_stream, number, operator.mul)

    @pytest.mark.parametrize("number", numbers)
    def test_div_numbers(self, waveframe_from_stream, number):
        """ Tests for adding numbers. """
        self._generic_op_test(waveframe_from_stream, number, operator.truediv)


class TestEqualityCheck:
    """ Tests for comparing waveframes. """

    def test_basic_comparison(self, waveframe_from_stream):
        """ Tests for equality checks which should return True. """
        wf1 = waveframe_from_stream
        assert wf1 == wf1
        wf2 = wf1.copy()
        assert wf2 == wf1

    def test_compare_non_waveframes(self, waveframe_from_stream):
        """ Tests for comparing objects which are not waveframes. """
        wf1 = waveframe_from_stream
        assert wf1 != 1
        assert wf1 != "bob"
        assert not wf1.equals(TestEqualityCheck)

    def test_not_equal(self, waveframe_from_stream):
        """ simple tests for waveframes which should not be equal. """
        wf1 = waveframe_from_stream
        wf2 = wf1 + 20
        assert wf1 != wf2

    def test_stats_not_equal(self, waveframe_from_stream):
        """
        When data are equal but stats are not the wfs should not be equal.
        """
        wf2 = waveframe_from_stream.copy()
        wf2["station"] = ""
        waveframe_from_stream.equals(wf2)
        assert waveframe_from_stream != wf2

    def test_stats_intersection(self, waveframe_from_stream):
        """
        If only the intersection of stats are used wf can still be equal.
        """
        wf1, wf2 = waveframe_from_stream, waveframe_from_stream.copy()
        wf2["new_col"] = "new_column"
        # these should now be not equal
        assert not wf1.equals(wf2, stats_intersection=False)
        # but if only using intersection of stats columns they are equal
        wf1.equals(wf2, stats_intersection=True)
        assert wf1.equals(wf2, stats_intersection=True)


class TestToFromStream:
    """ Tests for converting a stream to a WaveFrame. """

    def test_type(self, waveframe_from_stream):
        assert isinstance(waveframe_from_stream, WaveFrame)

    def test_to_stream(self, waveframe_from_stream, st_no_response):
        st = waveframe_from_stream.to_stream()
        assert isinstance(st, obspy.Stream)
        assert st == st_no_response


class TestDropNa:
    """ tests for dropping null values. """

    def _make_nan_wf(
        self,
        wf,
        y_inds: Union[int, slice] = slice(None),
        x_inds: Union[int, slice] = slice(None),
    ):
        """ make a waveframe with NaN values. """
        df = wf._df.copy()
        df.loc[y_inds, ("data", x_inds)] = np.NaN
        return WaveFrame(df)

    def test_drop_nan_column_all(self, waveframe_from_stream):
        """ Tests for dropping a column with all NaN. """
        wf = self._make_nan_wf(waveframe_from_stream, x_inds=0)
        # first test drops based on rows, this should drop all rows
        wf2 = wf.dropna(1, how="any")
        assert wf2 is not wf
        # there should no longer be any NaN
        assert not wf2.data.isnull().any().any()
        # the start of the columns should be 0
        assert wf2.data.columns[0] == 0
        # the starttime should have been updated
        assert (wf2["starttime"] > wf["starttime"]).all()
        # dropping using the all keyword should also work
        assert wf.dropna(1, how="all") == wf2

    def test_drop_nan_column_any(self, waveframe_from_stream):
        """ Tests for dropping a column with one NaN. """
        wf = self._make_nan_wf(waveframe_from_stream, 0, 0)
        # since only one value is NaN using how==all does nothing
        assert wf == wf.dropna(1, how="all")
        # but how==any should
        wf2 = wf.dropna(1, how="any")
        assert (wf["starttime"] < wf2["starttime"]).all()
        # the first index should always be 0
        assert wf2.data.columns[0] == 0

    def test_drop_nan_row_all(self, waveframe_from_stream):
        """ tests for dropping a row with all NaN"""
        wf = self._make_nan_wf(waveframe_from_stream, y_inds=0)
        wf2 = wf.dropna(0, how="all")
        assert wf2 == wf.dropna(0, how="any")
        # starttimes should not have changed
        assert (wf["starttime"][1:] == wf2["starttime"]).all()

    def test_drop_nan_row_any(self, waveframe_from_stream):
        """ test for dropping a row with one NaN. """
        wf = self._make_nan_wf(waveframe_from_stream, y_inds=0, x_inds=0)
        wf2 = wf.dropna(0, how="any")
        wf3 = wf.dropna(0, how="all")
        assert len(wf3) > len(wf2)

    def test_drop_all(self, waveframe_from_stream):
        """ tests for when all rows are dropped. """
        wf = self._make_nan_wf(waveframe_from_stream, x_inds=0)
        wf2 = wf.dropna(0, how="any")
        assert len(wf2) == 0


class TestStride:
    """ Tests for stridding data. """

    def test_overlap_gt_window_len_raises(self, waveframe_from_stream):
        """ Stride should rasie if the overlap is greater than window len. """
        wf = waveframe_from_stream
        with pytest.raises(ValueError):
            wf.stride(10, 100)

    def test_empty(self, waveframe_from_stream):
        """ Ensure striding works. """
        # Stridding with now input params should return a copy of waveframe.
        out = waveframe_from_stream.stride()
        assert isinstance(out, WaveFrame)
        assert out == waveframe_from_stream

    def test_overlap_default_window_len(self, waveframe_from_stream):
        """ Ensure strides can be overlapped. """
        wf = waveframe_from_stream
        # An overlap with the default window_len should also return a copy.
        wf2 = wf.stride(overlap=10)
        assert wf == wf2

    def test_no_overlap_half_len(self, waveframe_from_stream):
        """ ensure the stride when len is half creates a waveframe with 2x rows."""
        window_len = 1_500
        wf = waveframe_from_stream
        out = wf.stride(window_len=window_len)
        assert len(out) == 2 * len(wf)
        assert out.shape[-1] == window_len
        # starttimes and endtime should have been updated
        starttimes, endtimes = out["starttime"], out["endtime"]
        delta = out["delta"]
        data_len = out.shape[-1]
        assert starttimes[0] + data_len * delta[0] == starttimes[1]
        assert (endtimes - starttimes == (data_len - 1) * delta).all()
        assert endtimes[0] == starttimes[1] - delta[1]
        assert endtimes[0] + data_len * delta[0] == endtimes[1]


class TestResetIndex:
    """ tests for resetting index of waveframe. """


class TestSetIndex:
    """ Tests for setting index """
