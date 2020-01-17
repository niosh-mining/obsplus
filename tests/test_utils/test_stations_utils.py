"""
Tests for station utilities.
"""
import numpy as np
import obspy
import pandas as pd
import pytest

import obsplus
from obsplus.constants import NSLC, DF_TO_INV_COLUMNS
from obsplus.utils.stations import df_to_inventory


class TestDfToInventory:
    """ Tests for converting a dataframe to an obspy Inventory. """

    @staticmethod
    def _assert_dates_are_utc_or_none(obj):
        """ assert the start_date and end_date are UTC instances or None """
        start = getattr(obj, "start_date", None)
        end = getattr(obj, "end_date", None)
        for attr in [start, end]:
            assert isinstance(attr, obspy.UTCDateTime) or attr is None

    @pytest.fixture
    def df_from_inv(self):
        """ convert the default inventory to a df and return. """
        inv = obspy.read_inventory()
        return obsplus.stations_to_df(inv)

    @pytest.fixture
    def inv_from_df(self, df_from_inv):
        """ convert the station df back into an inventory. """
        return df_to_inventory(df_from_inv)

    @pytest.fixture
    def df_from_inv_from_df(self, inv_from_df):
        """ Is this getting confusing yet?"""
        return obsplus.stations_to_df(inv_from_df)

    @pytest.fixture
    def dummy_df(self):
        """ df 'inventory' with odd data types for the NSLC columns """
        numerics = [1.0, 1.0, 1.0, 1.0, 40.0, -111.0, 2000, 0, 250]
        row1 = numerics + ["2019-01-01", "2020-01-01"]
        row2 = numerics + ["2020-01-01", "2200-01-01"]
        return pd.DataFrame([row1, row2], columns=DF_TO_INV_COLUMNS)

    @pytest.fixture
    def df_with_response(self):
        """
        Add response information to the dataframe.

        Note: The datalogger is probably not correct for the test station.
        It is just to ensure the NRL can be used to get a response.
        """
        _inv = obsplus.load_dataset("bingham").station_client.get_stations()
        inv = _inv.select(station="NOQ")
        df = obsplus.stations_to_df(inv)

        # set instrument str
        sensor_keys = ("Nanometrics", "Trillium 120 Horizon")
        # get digitizer keys
        datalogger_keys = (
            "Nanometrics",
            "Centaur",
            "1 Vpp (40)",
            "Off",
            "Linear phase",
            "100",
        )
        # keep one as a tuple and convert the other to str
        df["sensor_keys"] = [sensor_keys for _ in range(len(df))]
        df["datalogger_keys"] = "__".join(datalogger_keys)
        # drop seed id
        df = df.drop(columns="seed_id")
        return df

    @pytest.fixture
    def df_with_partial_responses(self, df_with_response):
        """ test creating inv with partial responses. """
        # set one row to None
        df_with_response.loc[0, "sensor_keys"] = None
        return df_with_response

    def test_type(self, inv_from_df):
        """ An inv should have been returned. """
        assert isinstance(inv_from_df, obspy.Inventory)

    def test_column_dtypes(self, dummy_df):
        """ Make sure data types get set (particularly for NSLC columns) """
        inv = df_to_inventory(dummy_df)
        for network in inv.networks:
            assert network.code == "1"
            for station in network:
                assert station.code == "1"
            for channel in station:
                assert channel.code == "1"
                assert channel.location_code == "1"

    def test_new(self, df_from_inv, df_from_inv_from_df):
        """ Ensure the transformation is lossless from df side. """
        df1, df2 = df_from_inv, df_from_inv_from_df
        assert len(df1) == len(df2)
        assert set(df1.columns) == set(df2.columns)
        columns = sorted(list(df1.columns))
        df1 = df1[columns].sort_values(columns).reset_index(drop=True)
        df2 = df2[columns].sort_values(columns).reset_index(drop=True)
        assert df1.equals(df2)

    def test_dates_are_utc_datetime_objects(self, inv_from_df):
        """
        All the dates should be either None or instances of UTCDateTime.
        """
        for net in inv_from_df:
            for sta in net:
                self._assert_dates_are_utc_or_none(sta)
                for cha in sta:
                    self._assert_dates_are_utc_or_none(cha)

    def test_NaN_in_non_time_columns(self, df_from_inv):
        """
        If there are NaN values in non-time these should just be interp.
        as None.
        """
        df_from_inv.loc[2, "dip"] = np.NaN
        df_from_inv.loc[3, "azimuth"] = np.NaN
        # convert to inv
        inv = df_to_inventory(df_from_inv)
        # make sure dip is None
        dip_row = df_from_inv.loc[2]
        kwargs = {x: getattr(dip_row, x) for x in NSLC}
        inv_sub = inv.get_stations(**kwargs)
        assert inv_sub[0][0][0].dip is None
        # make sure azimuth is None
        dip_row = df_from_inv.loc[3]
        kwargs = {x: getattr(dip_row, x) for x in NSLC}
        inv_sub = inv.get_stations(**kwargs)
        assert inv_sub[0][0][0].azimuth is None

    @pytest.mark.requires_network
    def test_response(self, df_with_response):
        """ Ensure the NRL is used to pull responses. """
        inv = df_to_inventory(df_with_response)
        assert isinstance(inv, obspy.Inventory)
        for net in inv:
            for sta in net:
                for chan in sta:
                    assert chan.response is not None

    @pytest.mark.requires_network
    def test_response_one_missing(self, df_with_partial_responses):
        """ Ensure responses which can be got are fetched. """
        df = df_with_partial_responses
        inv = df_to_inventory(df_with_partial_responses)

        missing = df["sensor_keys"].isnull() | df["datalogger_keys"].isnull()
        missing_chan_data = set(df[missing]["channel"])

        for net in inv:
            for sta in net:
                for chan in sta:
                    name = chan.code
                    assert chan.response is not None or name in missing_chan_data

    def test_make_station_level_inventory(self, df_from_inv):
        """ Ensure station level invs can be constructed. """
        df = df_from_inv.drop(columns="channel")
        inv = df_to_inventory(df)
        for net in inv:
            for sta in net:
                assert not sta.channels, "there should be no channels"

    def test_make_network_level_inventory(self, df_from_inv):
        """ Ensure station level invs can be constructed. """
        df = df_from_inv.drop(columns=["channel", "station"])
        inv = df_to_inventory(df)
        for net in inv:
            assert not net.stations
