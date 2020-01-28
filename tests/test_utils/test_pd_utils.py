"""
Tests for the pandas utilites.
"""
import numpy as np
import obspy
import pandas as pd
import pytest

import obsplus
import obsplus.utils.pd as upd
from obsplus.utils.time import to_datetime64, to_timedelta64
from obsplus.constants import NSLC
from obsplus.exceptions import DataFrameContentError


@pytest.fixture
def simple_df():
    """ Return a simple dataframe. """
    cat = obsplus.load_dataset("bingham_test").event_client.get_events()
    df = obsplus.events_to_df(cat)
    return df


@pytest.fixture
def waveform_df():
    """ Create a dataframe with the basic required columns. """
    st = obspy.read()
    cols = list(NSLC) + ["starttime", "endtime"]
    df = pd.DataFrame([tr.stats for tr in st])[cols]
    df["starttime"] = to_datetime64(df["starttime"])
    df["endtime"] = to_datetime64(df["endtime"])
    return df


class TestApplyFuncsToColumns:
    """ Test applying functions to various columns. """

    def test_basic(self, simple_df):
        """ Ensure the functions are applied to the dataframe """
        df = simple_df
        td = np.timedelta64(1, "s")
        funcs = {"time": lambda x: x + td, "latitude": lambda x: x + 1}
        out = upd.apply_funcs_to_columns(simple_df, funcs)
        # a dataframe copy should have been returned
        assert out is not df
        # very the functions were applied
        assert (out["time"] == df["time"] + td).all()
        assert (out["latitude"] == df["latitude"] + 1).all()

    def test_skips_missing_column(self, simple_df):
        """ A function should not be called on a non-existent column. """
        funcs = {"bob": lambda x: x / 0}
        out = upd.apply_funcs_to_columns(simple_df, funcs)
        assert "bob" not in out.columns

    def test_inplace(self, simple_df):
        """ Inplace should be, well, inplace. """
        funcs = {"latitude": lambda x: x + 1}
        out = upd.apply_funcs_to_columns(simple_df, funcs, inplace=True)
        assert out is simple_df


class TestCastDtypes:
    """ tests for apply different datatypes to columns. """

    @pytest.fixture
    def time_df(self):
        """ Create a simple dataframe for converting to time/time delta"""
        df = pd.DataFrame(index=range(3), columns=["time", "delta"])
        # populate various time formats
        df.loc[0, "time"] = "2012-01-12T10:10:02"
        df.loc[1, "time"] = 120222002
        df.loc[2, "time"] = 100000
        # populate time delta columns
        df.loc[0, "delta"] = 0
        df.loc[1, "delta"] = 1
        df.loc[2, "delta"] = 3
        return df

    def test_basic(self, simple_df):
        """ simple test for casting datatypes. """
        out = upd.cast_dtypes(simple_df, {"time": str})
        assert all([isinstance(x, str) for x in out["time"]])

    def test_inplace(self, simple_df):
        out = upd.cast_dtypes(simple_df, {"latitude": int}, inplace=True)
        assert out is simple_df
        out2 = upd.cast_dtypes(simple_df, {"longitude": int})
        assert out2 is not simple_df

    def test_time_dtype(self, time_df):
        """ Test time dtype. """
        out1 = upd.cast_dtypes(time_df, {"time": "ops_datetime"})["time"]
        out2 = to_datetime64(time_df["time"])
        assert (out1 == out2).all()

    def test_time_delta(self, time_df):
        """ Test that timedelta dtype. """
        out1 = upd.cast_dtypes(time_df, {"delta": "ops_timedelta"})["delta"]
        out2 = to_timedelta64(time_df["delta"])
        assert (out1 == out2).all()

    def test_utc_datetime(self, time_df):
        """ Tests for converting to UTCDateTime. """
        out = upd.cast_dtypes(time_df, {"time": "utcdatetime"})
        assert all([isinstance(x, obspy.UTCDateTime) for x in out["time"]])

    def test_empty_with_columns(self):
        """ An empty dataframe should still have the datatypes castll. """
        # get columns and dtypes
        columns = ["time", "space"]
        dtypes = {"time": "ops_datetime", "space": float}
        # create 2 dfs, one empty one with values
        df_empty = pd.DataFrame(columns=columns)
        df_full = pd.DataFrame([[1, 1.2]], columns=columns)
        # run cast_dtypes on both and compare
        out_empty = upd.cast_dtypes(df_empty, dtype=dtypes).dtypes
        out_full = upd.cast_dtypes(df_full, dtype=dtypes).dtypes
        assert (out_empty == out_full).all()

    def test_empty_no_columns(self):
        """ Ensure empty dataframes with no columns just returns. """
        df = pd.DataFrame()
        out = upd.cast_dtypes(df, dtype={"bob": int})
        assert isinstance(out, pd.DataFrame)
        assert out.empty
        assert len(out.columns) == 0


class TestGetWaveformsBulkArgs:
    """
    Tests for getting the bulk arguments for get_waveforms_bulk.
    """

    def assert_wellformed_bulk_args(self, bulk_args):
        """ Assert the bulk args are as expected. """
        for bulk_arg in bulk_args:
            assert len(bulk_arg) == 6
            for str_thing in bulk_arg[:4]:
                assert isinstance(str_thing, str)
            for date_thing in bulk_arg[4:]:
                assert isinstance(date_thing, obspy.UTCDateTime)

    def test_basic_get_nslc(self, waveform_df):
        """ Test bulk args with no only required columns. """
        bulk_args = upd.get_waveforms_bulk_args(waveform_df)
        self.assert_wellformed_bulk_args(bulk_args)

    def test_missing_column_raises(self, waveform_df):
        """ Test that a missing required column raises. """
        df = waveform_df.drop(columns="starttime")
        with pytest.raises(DataFrameContentError):
            upd.get_waveforms_bulk_args(df)

    def test_missing_value_raises(self, waveform_df):
        """ Ensure any NaN values raises. """
        waveform_df.loc[0, "starttime"] = np.NaN
        with pytest.raises(DataFrameContentError):
            upd.get_waveforms_bulk_args(waveform_df)

    def test_extractra_columns_work(self, waveform_df):
        """ Extra columns shouldn't effect anything. """
        waveform_df["bob"] = 10
        bulk = upd.get_waveforms_bulk_args(waveform_df)
        self.assert_wellformed_bulk_args(bulk)

    def test_bad_start_endtime_raises(self, waveform_df):
        """ If any starttime is before endtime it should raise. """
        td = np.timedelta64(10, "s")
        df = waveform_df.copy()
        df["starttime"] = df["endtime"] + td
        with pytest.raises(DataFrameContentError):
            upd.get_waveforms_bulk_args(df)

    def test_bad_seed_id_raises(self, waveform_df):
        """ Seed ids cannot (currently) have wildcards """
        waveform_df.loc[0, "network"] = "B?"
        with pytest.raises(DataFrameContentError):
            upd.get_waveforms_bulk_args(waveform_df)

    def test_enddate_startdate(self, waveform_df):
        """
        enddate and startdate are also valid column names for starttime
        endtime. These occur when getting data from inventories.
        """
        rename = dict(starttime="startdate", endtime="enddate")
        df = waveform_df.rename(columns=rename)
        bulk = upd.get_waveforms_bulk_args(df)
        assert len(bulk) == len(df)
        self.assert_wellformed_bulk_args(bulk)


class TestMisc:
    """ Misc. small tests. """

    def test_replace_or_shallow_none(self, waveform_df):
        out = upd.replace_or_swallow(waveform_df, None)
        assert out.equals(waveform_df)
