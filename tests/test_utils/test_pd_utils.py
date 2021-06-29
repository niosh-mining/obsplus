"""
Tests for the pandas utilites.
"""
import numpy as np

import obspy
import pandas as pd
import pytest

import obsplus
import obsplus.utils.pd as upd
from obsplus.constants import NSLC
from obsplus.exceptions import DataFrameContentError
from obsplus.utils.time import to_datetime64, to_timedelta64
from obsplus.utils.pd import loc_by_name, get_index_group, expand_loc


@pytest.fixture
def simple_df():
    """Return a simple dataframe."""
    cat = obsplus.load_dataset("bingham_test").event_client.get_events()
    df = obsplus.events_to_df(cat)
    return df


@pytest.fixture
def waveform_df():
    """Create a dataframe with the basic required columns."""
    st = obspy.read()
    cols = list(NSLC) + ["starttime", "endtime"]
    df = pd.DataFrame([tr.stats for tr in st])[cols]
    df["starttime"] = to_datetime64(df["starttime"])
    df["endtime"] = to_datetime64(df["endtime"])
    return df


class TestApplyFuncsToColumns:
    """Test applying functions to various columns."""

    def test_basic(self, simple_df):
        """Ensure the functions are applied to the dataframe"""
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
        """A function should not be called on a non-existent column."""
        funcs = {"bob": lambda x: x / 0}
        out = upd.apply_funcs_to_columns(simple_df, funcs)
        assert "bob" not in out.columns

    def test_inplace(self, simple_df):
        """Inplace should be, well, inplace."""
        funcs = {"latitude": lambda x: x + 1}
        out = upd.apply_funcs_to_columns(simple_df, funcs, inplace=True)
        assert out is simple_df


class TestCastDtypes:
    """tests for apply different datatypes to columns."""

    @pytest.fixture
    def time_df(self):
        """Create a simple dataframe for converting to time/time delta"""
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
        """simple test for casting datatypes."""
        out = upd.cast_dtypes(simple_df, {"time": str})
        assert all([isinstance(x, str) for x in out["time"]])

    def test_time_dtype(self, time_df):
        """Test time dtype."""
        out1 = upd.cast_dtypes(time_df, {"time": "ops_datetime"})["time"]
        out2 = to_datetime64(time_df["time"])
        assert (out1 == out2).all()

    def test_time_delta(self, time_df):
        """Test that timedelta dtype."""
        out1 = upd.cast_dtypes(time_df, {"delta": "ops_timedelta"})["delta"]
        out2 = to_timedelta64(time_df["delta"])
        assert (out1 == out2).all()

    def test_utc_datetime(self, time_df):
        """Tests for converting to UTCDateTime."""
        out = upd.cast_dtypes(time_df, {"time": "utcdatetime"})
        assert all([isinstance(x, obspy.UTCDateTime) for x in out["time"]])

    def test_empty_with_columns(self):
        """An empty dataframe should still have the datatypes cast."""
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
        """Ensure empty dataframes with no columns just returns."""
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
        """Assert the bulk args are as expected."""
        for bulk_arg in bulk_args:
            assert len(bulk_arg) == 6
            for str_thing in bulk_arg[:4]:
                assert isinstance(str_thing, str)
            for date_thing in bulk_arg[4:]:
                assert isinstance(date_thing, obspy.UTCDateTime)

    def test_basic_get_nslc(self, waveform_df):
        """Test bulk args with no only required columns."""
        bulk_args = upd.get_waveforms_bulk_args(waveform_df)
        self.assert_wellformed_bulk_args(bulk_args)

    def test_missing_column_raises(self, waveform_df):
        """Test that a missing required column raises."""
        df = waveform_df.drop(columns="starttime")
        with pytest.raises(DataFrameContentError):
            upd.get_waveforms_bulk_args(df)

    def test_missing_value_raises(self, waveform_df):
        """Ensure any NaN values raises."""
        waveform_df.loc[0, "starttime"] = np.NaN
        with pytest.raises(DataFrameContentError):
            upd.get_waveforms_bulk_args(waveform_df)

    def test_extractra_columns_work(self, waveform_df):
        """Extra columns shouldn't effect anything."""
        waveform_df["bob"] = 10
        bulk = upd.get_waveforms_bulk_args(waveform_df)
        self.assert_wellformed_bulk_args(bulk)

    def test_bad_start_endtime_raises(self, waveform_df):
        """If any starttime is before endtime it should raise."""
        td = np.timedelta64(10, "s")
        df = waveform_df.copy()
        df["starttime"] = df["endtime"] + td
        with pytest.raises(DataFrameContentError):
            upd.get_waveforms_bulk_args(df)

    def test_bad_seed_id_raises(self, waveform_df):
        """Seed ids cannot (currently) have wildcards"""
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


class TestGetSeedIdSeries:
    """
    Tests for getting seed id series from dataframes with  network, station,
    location, channel columns.
    """

    @pytest.fixture(scope="class")
    def pick_df(self):
        """Return the pick dataframe of Bingham."""
        ds = obsplus.load_dataset("bingham_test")
        cat = ds.event_client.get_events()
        return obsplus.picks_to_df(cat)

    def test_seed_id_basic(self, pick_df):
        """Standard usage."""
        seed = upd.get_seed_id_series(pick_df)
        assert (seed == pick_df["seed_id"]).all()

    def test_bad_subset(self, pick_df):
        """Bad subset should raise valuerror."""
        with pytest.raises(ValueError):
            upd.get_seed_id_series(pick_df, subset=["network", "monkey"])

    def test_dataframe_missing_columns(self, pick_df):
        """Dataframe without required columns should raise ValueError."""
        new = pick_df.drop(columns=["network", "location"])
        with pytest.raises(ValueError):
            upd.get_seed_id_series(new)
        # But it should work if only the required subset is there
        out = upd.get_seed_id_series(new, subset=["station", "channel"])
        assert len(out) == len(pick_df)
        split = out.str.split(".", expand=True)
        assert (split[0] == pick_df["station"]).all()
        assert (split[1] == pick_df["channel"]).all()

    def test_one_subset_raises(self, pick_df):
        """At least two columns are required in subset."""
        with pytest.raises(ValueError):
            upd.get_seed_id_series(pick_df, subset=["network"])


class TestLocByName:
    """Tests for selecting values on dataframe using various levels of index."""

    def test_nameless_index_raises(self, bingham_events_df):
        """A nameless index is not permitted for this function."""
        with pytest.raises(KeyError, match="with named indices"):
            loc_by_name(bingham_events_df)

    def test_wrong_names(self, bingham_events_df):
        """Ensure an error message is raised if wrong index names are used."""
        df = bingham_events_df.set_index(["event_id", "magnitude"])
        with pytest.raises(KeyError, match="names are not in the df index"):
            loc_by_name(df, amplitude=1)

    def test_slice_only_level(self, bingham_events_df):
        """Ensure method works with only level, should be equivalent to loc"""
        df = bingham_events_df.set_index("event_id")
        first_ids = df.index.values[:3]
        out1 = df.loc[first_ids]
        out2 = loc_by_name(df, event_id=first_ids)
        assert out1.equals(out2)

    def test_slice_one_level(self, bingham_events_df):
        """Ensure method works slicing on one level."""
        df = bingham_events_df.set_index(["event_id", "time", "magnitude"]).sort_index()
        out = loc_by_name(df, magnitude=slice(2, None))
        expected = bingham_events_df[bingham_events_df["magnitude"] > 2]
        assert len(out) == len(expected)
        eid1 = out.reset_index()["event_id"].iloc[0]
        eid2 = expected["event_id"].iloc[0]
        assert eid1 == eid2

    def test_slice_all_levels(self, bingham_events_df):
        """Ensure slicing works on multiple levels."""
        old = bingham_events_df
        df = old.set_index(["event_id", "latitude", "depth"]).sort_index()
        eids = old["event_id"].iloc[:5]
        lats = old["latitude"]

        out = loc_by_name(
            df,
            depth=slice(0, None),
            event_id=eids,
            latitude=slice(lats.min(), lats.max()),
        )
        expected = old[
            (lats >= lats.min())
            & (lats <= lats.max())
            & (old["event_id"].isin(eids))
            & (old["depth"] >= 0)
        ]
        assert len(expected) == len(out)
        assert set(expected["event_id"]) == set(out.reset_index()["event_id"])


class TestInIndexGroup:
    """Tests for getting indices from index or columns."""

    @pytest.fixture
    def df_one_ind(self):
        """A dataframe with a single index column"""
        df = pd.DataFrame(range(100), columns=["ind"])
        return df

    @pytest.fixture
    def df_multi_ind(self):
        """A dataframe with multiple groupby_columns."""
        col1 = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        col2 = [0, 1, 2, 0, 1, 2, 3, 0, 1, 2]
        out = pd.DataFrame(np.stack([col1, col2]).T, columns=["col1", "index"])
        return out

    def test_one_ind_positive(self, df_one_ind):
        """Ensure positive ints work."""
        out = get_index_group(df_one_ind, 0)
        assert np.all(out.values == df_one_ind.index.values[:0])

    def test_on_ind_negative(self, df_one_ind):
        """Tests for negative indices."""
        out = get_index_group(df_one_ind, -1)
        assert np.all(out.values == df_one_ind.index.values[-1:])

    def test_multiple_ind_positive(self, df_multi_ind):
        """Ensure algorithm works with multiple indicies."""
        out = get_index_group(df_multi_ind, 1, column_group=["col1"])
        vals = df_multi_ind.loc[out]["index"]
        assert np.all(vals == 1)

    def test_multiple_some_too_large(self, df_multi_ind):
        """test when some groups final values are exceeded."""
        out = get_index_group(df_multi_ind, 3, column_group=["col1"])
        assert np.all(out.values == 6)

    def test_multiple_negative(self, df_multi_ind):
        """tests multi-group negative index"""
        out = get_index_group(df_multi_ind, -1, column_group=["col1"])
        assert np.all(out == np.array([2, 6, 9]))

    def test_multiple_some_too_small(self, df_multi_ind):
        """Test for when negative index is too small."""
        out = get_index_group(df_multi_ind, -4, column_group=["col1"])
        assert np.all(out.values == np.array([3]))

    def test_empty(
        self,
    ):
        """Ensure empty df still works."""
        df = pd.DataFrame(columns=["index", "col1"])
        out = get_index_group(df, 0, column_group=["col1"])
        assert not len(out)


class TestExpandLoc:
    """Tests for expanding a selection of a dataframe."""

    def test_expand_loc_column(self):
        """Ensure expand_loc works on a column"""
        df = pd.DataFrame([1, 2, 3], columns=["col1"])
        values = [0, 1, 1, 2, 3]
        out = expand_loc(df, col1=values)
        assert len(out) == len(values)


class TestReplaceOrShallow:
    """Misc. small tests."""

    def test_replace_or_shallow_none(self, waveform_df):
        """Test when replace is non the dataframe is simply returned."""
        out = upd.replace_or_swallow(waveform_df, None)
        assert out.equals(waveform_df)
