"""
Tests for the pandas utilites.
"""
import numpy as np
import pandas as pd
import obsplus
import obsplus.utils.pd as upd
from obsplus.utils.time import to_datetime64, to_timedelta64
import pytest


@pytest.fixture
def simple_df():
    """ Return a simple dataframe. """
    cat = obsplus.load_dataset("bingham").event_client.get_events()
    df = obsplus.events_to_df(cat)
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


class TestApplyDtypes:
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
        out1 = upd.cast_dtypes(time_df, {"time": "time"})["time"]
        out2 = to_datetime64(time_df["time"])
        assert (out1 == out2).all()

    def test_time_delta(self, time_df):
        """ Test that timedelta dtype. """
        out1 = upd.cast_dtypes(time_df, {"delta": "timedelta"})["delta"]
        out2 = to_timedelta64(time_df["delta"])
        assert (out1 == out2).all()
