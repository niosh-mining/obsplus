"""
Tests for the WaveFrame class.
"""
import operator

import numpy as np
import obspy
import pandas as pd
import pytest
from obspy import UTCDateTime

from obsplus import WaveFrame
from obsplus.constants import NSLC
from obsplus.exceptions import DataFrameContentError
from obsplus.utils.misc import suppress_warnings


class Testbasics:
    """Basic tests of waveframe."""

    def test_get_item(self, stream_wf):
        """get item should return the series from the stats df."""
        wf = stream_wf
        ser = wf["station"]
        assert (ser == "RJOB").all()

    def test_set_item(self, stream_wf):
        """Set item should set a column in the stats dataframe."""
        wf = stream_wf
        wf["station"] = "JOB"
        assert (wf["station"] == "JOB").all()

    def test_add_new_column(self, stream_wf):
        """Add a new column to stats."""
        wf = stream_wf
        wf["new_col"] = "heyo"
        out = wf["new_col"]
        assert isinstance(out, pd.Series)
        assert len(out) == len(wf)

    def test_filtering(self, stream_wf):
        """Ensure passing a boolean filters the dataframe."""
        wf = stream_wf
        ser = wf["channel"].str.endswith("Z")
        wf2 = wf[ser]
        assert len(wf2) == 1
        assert wf2["channel"].str.endswith("Z").all()

    def test_data_and_stats_write_only(self, stream_wf):
        """The data and stats parameter should be write only."""
        data, stats = stream_wf.data, stream_wf.stats
        with pytest.raises(AttributeError):
            stream_wf.data = data
        with pytest.raises(AttributeError):
            stream_wf.stats = stats

    def test_cant_access_class_data_stats(self):
        """Class level data and stats should not be accessible."""
        with pytest.raises(AttributeError):
            WaveFrame.stats
        with pytest.raises(AttributeError):
            WaveFrame.data

    def test_set_readonly_stats(self, stream_wf):
        """Ensure an error is raised when accessing protected stats."""
        with pytest.raises(AttributeError):
            stream_wf["endtime"] = np.datetime64("2019-01-01")

    def test_helpful_message_on_access_non_existent_column(self, stream_wf):
        """
        Ensure a helpful message is raised when accessing a stats column
        which doesnt exist.
        """
        with pytest.raises(KeyError) as e:
            stream_wf["_this_isnt_a_column!!"]
        msg = str(e.value)
        assert "is not a stats column" in msg

    def test_set_new_column(self, stream_wf):
        """get_item syntax should allow for setting new values on index."""
        # a single value should work
        stream_wf["bob"] = 2
        assert (stream_wf["bob"] == 2).all()
        # as should an array
        ones = np.ones(len(stream_wf))
        stream_wf["ones"] = ones
        assert (stream_wf["ones"] == 1).all()
        # or a series with an index
        sub = stream_wf["delta"].loc[1:]
        stream_wf["sub1"] = sub
        sub1_out = stream_wf["sub1"]
        assert pd.isnull(sub1_out.loc[0])
        assert (sub1_out.loc[1:] == sub).all()


class TestConstructor:
    """Basic tests for creating WaveFrame instances."""

    t1 = UTCDateTime("2009-08-24T00-20-03")

    def _make_stats_df(self, args):
        names = list(NSLC) + ["starttime", "endtime"]
        df = pd.DataFrame([args], columns=names)
        return df

    def test_wildcard_raises(self):
        """WaveFrame does not support using wildcards in str params of bulk."""
        bad_nslc = ["*", "bob", "01", "BHZ", self.t1, self.t1 + 10]
        df = self._make_stats_df(bad_nslc)
        # since there is a wildcard it should raise a ValueError
        with pytest.raises(DataFrameContentError):
            WaveFrame(waveforms=obspy.read(), stats=df)

    def test_test_bad_starttime_endtime_raises(self):
        """Ensure bad starttime/endtimes raise."""
        bad_args = ["UU", "BOB", "01", "BHZ", self.t1, self.t1 - 10]
        df = self._make_stats_df(bad_args)
        # since starttime is after endtime this should raise
        with pytest.raises(DataFrameContentError):
            WaveFrame(waveforms=obspy.read(), stats=df)

    def test_null_values_raise(self):
        """Null values in any column should raise."""
        bad_args = ["UU", "BOB", "01", None, self.t1, self.t1 - 10]
        df = self._make_stats_df(bad_args)
        with pytest.raises(DataFrameContentError):
            WaveFrame(waveforms=obspy.read(), stats=df)

    def test_date_columns_renamed(self):
        """Ensure enddate and startdate get renamed to starttime and endtime"""
        bulk = ["BW", "RJOB", "", "EHZ", self.t1, self.t1 + 10]
        names = list(NSLC) + ["startdate", "enddate"]
        df = pd.DataFrame([bulk], columns=names)
        st = obspy.read()
        wf = WaveFrame(waveforms=st, stats=df)
        assert {"starttime", "endtime"}.issubset(set(wf.stats.columns))

    def test_gappy_traces(self, waveframe_gap):
        """Ensure gappy data still works."""

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

    def test_init_waveframe_from_waveframe(self, stream_wf):
        """A waveframe should be valid input to waveframe constructor"""
        wf1 = stream_wf
        wf2 = WaveFrame(wf1)
        assert wf1 is not wf2
        assert wf1._df is not wf2._df
        assert wf1 == wf2

    def test_init_waveframe_from_waveframe_df(self, stream_wf):
        """A waveframe can be inited from a dataframe from a waveframe."""
        wf1 = stream_wf
        wf2 = WaveFrame(wf1._df)
        assert wf1 is not wf2
        assert wf1._df is not wf2._df
        assert wf1 == wf2

    def test_init_waveframe_from_waveframe_parts(self, stream_wf):
        """A wavefrom should be init'able from a waveframes parts"""
        wf1 = stream_wf
        wf2 = WaveFrame(waveforms=wf1.data, stats=wf1.stats)
        assert wf1 is not wf2
        assert wf1._df is not wf2._df
        assert wf1 == wf2

    def test_waveframe_has_delta(self, stream_wf):
        """Waveframe should have a delta parameter in its stats."""
        stats = stream_wf.stats
        assert "delta" in stats.columns
        assert np.issubdtype(stats["delta"].values.dtype, np.timedelta64)

    def test_stream_uneven(self, st_no_response):
        """Tests for when streams are not evenly sized."""
        st = st_no_response
        st[0].data = st[0].data[:100]
        wf = WaveFrame.from_stream(st)
        # only 100 values should be Non-null
        assert (~wf.data.loc[0].isnull()).sum() == 100
        # the shape should still be greater than 100
        assert wf.data.shape[-1] > 100

    def test_from_stats(self, stream_wf):
        """Ensure a new wavefream from stats can be created."""
        stats = stream_wf.stats
        delta = np.timedelta64(10, "D")

        stats["starttime"] += delta
        stats["endtime"] += delta
        out = stream_wf.from_stats(stats)
        assert isinstance(out, WaveFrame)
        assert out != stream_wf
        assert (out["starttime"] == stats["starttime"]).all()

    def test_from_data(self, stream_wf):
        """Ensure a new waveframe can be created from a new data df."""
        data = (stream_wf + 10).data
        out = stream_wf.from_data(data)
        assert isinstance(out, WaveFrame)
        assert np.allclose(out.data.values, stream_wf.data.values + 10)


class TestValidate:
    """tests for validating waveframes."""

    def test_missing_required_columns_raises(self, stream_wf):
        """
        Ensure if any of the required stats columns are missing validate raises.
        """
        wf = stream_wf
        # drop one of the requried columns
        df = wf._df
        df.drop(columns=["delta"], level=1, inplace=True)
        with pytest.raises(AssertionError):
            wf.validate()

    def test_inconsistent_endtimes(self, stream_wf):
        """
        If the endtime is not consistent with the starttime + len of data raise.
        """
        df = stream_wf._df
        df.loc[:, ("stats", "endtime")] += df.loc[:, ("stats", "delta")] * 10
        with pytest.raises(AssertionError):
            stream_wf.validate()

    def test_validation_report(self, stream_wf):
        """Ensure a report can be returned."""
        wf = stream_wf
        df = wf.validate(report=True)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


class TestBasicOperations:
    """Tests for basic operations."""

    numbers = (0, 1, -1, 1.11, 10e2, 1 + 10j)

    def _generic_op_test(self, wf1, number, op):
        """Apply generic operation tests."""
        # div 0 raises runtime warning, this is ok.
        with suppress_warnings():
            wf2 = op(wf1, number)
            data1 = op(wf1.data.values, number)

        assert isinstance(wf2, WaveFrame)
        assert wf2 is not wf1
        data2 = wf2.data.values
        close = np.isclose(data1, data2)
        not_finite = ~np.isfinite(data1)
        assert np.all(close | not_finite)

    @pytest.mark.parametrize("number", numbers)
    def test_add_numbers(self, stream_wf, number):
        """Tests for adding numbers."""
        self._generic_op_test(stream_wf, number, operator.add)

    @pytest.mark.parametrize("number", numbers)
    def test_subtract_numbers(self, stream_wf, number):
        """Tests for adding numbers."""
        self._generic_op_test(stream_wf, number, operator.sub)

    @pytest.mark.parametrize("number", numbers)
    def test_mult_numbers(self, stream_wf, number):
        """Tests for adding numbers."""
        self._generic_op_test(stream_wf, number, operator.mul)

    @pytest.mark.parametrize("number", numbers)
    def test_div_numbers(self, stream_wf, number):
        """Tests for adding numbers."""
        self._generic_op_test(stream_wf, number, operator.truediv)


class TestEqualityCheck:
    """Tests for comparing waveframes."""

    def test_basic_comparison(self, stream_wf):
        """Tests for equality checks which should return True."""
        wf1 = stream_wf
        assert wf1 == wf1
        wf2 = wf1.copy()
        assert wf2 == wf1

    def test_compare_non_waveframes(self, stream_wf):
        """Tests for comparing objects which are not waveframes."""
        wf1 = stream_wf
        assert wf1 != 1
        assert wf1 != "bob"
        assert not wf1.equals(TestEqualityCheck)

    def test_not_equal(self, stream_wf):
        """simple tests for waveframes which should not be equal."""
        wf1 = stream_wf
        wf2 = wf1 + 20
        assert wf1 != wf2

    def test_stats_not_equal(self, stream_wf):
        """
        When data are equal but stats are not the wfs should not be equal.
        """
        wf2 = stream_wf.copy()
        wf2["station"] = ""
        stream_wf.equals(wf2)
        assert stream_wf != wf2

    def test_stats_intersection(self, stream_wf):
        """
        If only the intersection of stats are used wf can still be equal.
        """
        wf1, wf2 = stream_wf, stream_wf.copy()
        wf2["new_col"] = "new_column"
        # these should now be not equal
        assert not wf1.equals(wf2, stats_intersection=False)
        # but if only using intersection of stats columns they are equal
        wf1.equals(wf2, stats_intersection=True)
        assert wf1.equals(wf2, stats_intersection=True)
