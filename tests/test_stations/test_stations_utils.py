"""
Tests for station utilities.
"""
import obspy
import pytest

import obsplus
from obsplus.stations.utils import df_to_inventory


class TestDfToInventory:
    """ Tests for converting a dataframe to an obspy Inventory. """

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

    def test_type(self, inv_from_df):
        """ An inv should have been returned. """
        assert isinstance(inv_from_df, obspy.Inventory)

    def test_new(self, df_from_inv, df_from_inv_from_df):
        """ Ensure the transformation is lossless from df side. """
        df1, df2 = df_from_inv, df_from_inv_from_df
        assert len(df1) == len(df2)
        assert set(df1.columns) == set(df2.columns)
        columns = sorted(list(df1.columns))
        df1 = df1[columns].sort_values(columns).reset_index(drop=True)
        df2 = df2[columns].sort_values(columns).reset_index(drop=True)
        assert df1.equals(df2)
