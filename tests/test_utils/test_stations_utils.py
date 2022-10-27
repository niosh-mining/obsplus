"""
Tests for station utilities.
"""
import numpy as np
import obspy
import pandas as pd
import pytest
import json

import obsplus
from obsplus.constants import NSLC, DF_TO_INV_COLUMNS
from obsplus.utils import get_seed_id_series
from obsplus.utils.misc import suppress_warnings
from obsplus.utils.stations import df_to_inventory, get_station_client
from obsplus.utils.time import to_utc
from obsplus.interfaces import StationClient
from obsplus.exceptions import AmbiguousResponseError


class TestDfToInventory:
    """Tests for converting a dataframe to an obspy Inventory."""

    @staticmethod
    def _assert_dates_are_utc_or_none(obj):
        """assert the start_date and end_date are UTC instances or None"""
        start = getattr(obj, "start_date", None)
        end = getattr(obj, "end_date", None)
        for attr in [start, end]:
            assert isinstance(attr, obspy.UTCDateTime) or attr is None

    @pytest.fixture
    def df_from_inv(self):
        """convert the default inventory to a df and return."""
        inv = obspy.read_inventory()
        return obsplus.stations_to_df(inv)

    @pytest.fixture
    def inv_from_df(self, df_from_inv):
        """convert the station df back into an inventory."""
        out = df_to_inventory(df_from_inv)
        return out

    @pytest.fixture
    def df_from_inv_from_df(self, inv_from_df):
        """Is this getting confusing yet?"""
        return obsplus.stations_to_df(inv_from_df)

    @pytest.fixture(params=[(1.0, "01"), (1, "01"), ("1", "1"), ("01", "01")])
    def nslc_dtype_variation(self, request):
        """
        Return a dataframe with different datatypes for the nslc information
        """
        common = [
            request.param[0],
            request.param[0],
            request.param[0],
            request.param[0],
            40.0,
            -111.0,
            2000,
            0,
            250,
        ]
        row1 = common + ["2019-01-01", "2020-01-01"]
        row2 = common + ["2020-01-01", "2020-01-01"]
        return pd.DataFrame([row1, row2], columns=DF_TO_INV_COLUMNS), request.param[1]

    @pytest.fixture(params=range(4))
    def invalid_nslc(self, request):
        """
        Return a dataframes with bad data types for the NSLC data
        """
        common = ["a", "b", "c", "d", 50.0, -111.0, 2000, 0, 250]
        common[request.param] = "1.0"
        row1 = common + ["2019-01-01", "2020-01-01"]
        row2 = common + ["2020-01-01", "2020-01-01"]
        return pd.DataFrame([row1, row2], columns=DF_TO_INV_COLUMNS)

    @pytest.fixture
    def inv_df_duplicate_channels(self, df_from_inv):
        """
        Create a dataframe with duplicate channels that have different
        start/end dates.
        """
        # first add duplicates of fur with different start/end times
        df_from_inv["end_date"] = np.datetime64("2020-01-01")
        sub_fur = df_from_inv[df_from_inv["station"] == "FUR"]
        sub_fur["end_date"] = sub_fur["start_date"] - np.timedelta64(1, "Y")
        sub_fur["start_date"] = sub_fur["end_date"] - np.timedelta64(3, "Y")
        new_df = pd.concat([df_from_inv, sub_fur], ignore_index=True).reset_index(
            drop=True
        )
        return new_df

    def test_type(self, inv_from_df):
        """An inv should have been returned."""
        assert isinstance(inv_from_df, obspy.Inventory)

    def test_nslc_variations_float(self, nslc_dtype_variation):
        """Make sure data types get set (particularly for NSLC columns)"""
        inp = nslc_dtype_variation[0]
        expect = nslc_dtype_variation[1]
        inv = df_to_inventory(inp)
        for network in inv.networks:
            assert network.code == expect
            for station in network:
                assert station.code == expect
                for channel in station:
                    assert channel.code == expect
                    assert channel.location_code == expect

    def test_invalid_nslc(self, invalid_nslc):
        """Make sure data types get set (particularly for NSLC columns)"""
        with pytest.raises(TypeError, match="cannot contain '.'"):
            df_to_inventory(invalid_nslc)

    def test_new(self, df_from_inv, df_from_inv_from_df):
        """Ensure the transformation is lossless from df side."""
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

    def test_make_station_level_inventory(self, df_from_inv):
        """Ensure station level invs can be constructed."""
        df = df_from_inv.drop(columns="channel")
        inv = df_to_inventory(df)
        for net in inv:
            for sta in net:
                assert not sta.channels, "there should be no channels"

    def test_make_network_level_inventory(self, df_from_inv):
        """Ensure station level invs can be constructed."""
        df = df_from_inv.drop(columns=["channel", "station"])
        inv = df_to_inventory(df)
        for net in inv:
            assert not net.stations

    def test_00_location_code(self, df_from_inv):
        """Ensure a 00 location code makes it into the inventory."""
        df = df_from_inv.copy()
        df["location"] = "00"
        inv = df_to_inventory(df)
        for channel in inv.get_contents()["channels"]:
            assert channel.split(".")[2] == "00"

    def test_duplicate_stations(self, inv_df_duplicate_channels):
        """
        Ensure duplicate stations create Station objects with correct
        time range.
        """
        df = inv_df_duplicate_channels
        fur_df = df[df["station"] == "FUR"]
        inv = df_to_inventory(fur_df).select(station="FUR")
        stations = inv.networks
        assert len(stations) == 1
        fur = stations[0]
        assert fur.start_date == to_utc(fur_df["start_date"].min())
        assert fur.end_date == to_utc(fur_df["end_date"].max())

    def test_unsupported_column_warns(self, df_from_inv):
        """Ensure an unsupported column issues a warning."""
        df_from_inv["bob"] = 1
        with pytest.warns(UserWarning) as w:
            df_to_inventory(df_from_inv)
        messages = " ".join([x.message.args[0] for x in w.list])
        assert "found unexpected columns" in messages


@pytest.mark.requires_network
class TestDfToInventoryGetResponses:
    """Tests for getting responses with the df_to_inventory function."""

    def has_valid_response(self, inventory, expected_missing=None):
        """
        Return True if the inventory has expected responses.
        Ensure expected_missing are empty, if a list of seed ids is provided.
        """
        assert isinstance(inventory, obspy.Inventory)
        expected_missing = {} if expected_missing is None else expected_missing
        for net in inventory:
            for sta in net:
                for chan in sta:
                    seed = f"{net.code}.{sta.code}.{chan.location_code}.{chan.code}"
                    if seed in expected_missing and chan.response is not None:
                        return False
                    if seed not in expected_missing and chan.response is None:
                        return False
        return True

    @pytest.fixture
    def df_with_nrl_response(self):
        """
        Add NRL response information to the dataframe.

        Note: The datalogger is probably not correct for the test station.
        It is just to ensure the NRL can be used to get a response.
        """
        _inv = obsplus.load_dataset("bingham_test").station_client.get_stations()
        inv = _inv.select(station="NOQ")

        with suppress_warnings():
            df = obsplus.stations_to_df(inv)

        # set instrument str
        sensor_keys = ("Nanometrics", "Trillium 120 Horizon", "Trillium 120 Horizon")
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
        df["datalogger_keys"] = json.dumps(datalogger_keys)
        # drop seed id
        df = df.drop(columns="seed_id")
        return df

    @pytest.fixture
    def df_with_get_stations_kwargs(self):
        """
        Add response information to the dataframe using get_stations_kwargs.

        Add an additional station which will need to get all data from other
        columns.
        """
        _inv = obsplus.load_dataset("bingham_test").station_client.get_stations()
        inv = _inv.select(station="NOQ")

        with suppress_warnings():
            df = obsplus.stations_to_df(inv).reset_index()

        # set get_station_kwargs for last two channels, leave first empty
        kwargs_list = [""]
        for ind, row in df.iloc[1:].iterrows():
            kwargs = {x: row[x] for x in NSLC}
            kwargs["endafter"] = str(to_utc(row["start_date"]))
            kwargs_list.append(kwargs)
        # set last kwargs to str to simulate reading from csv
        kwargs_list[-1] = str(kwargs_list[-1])
        df["get_station_kwargs"] = kwargs_list
        # set the first kwargs to a string to make sure it can be parsed
        # this is important for eg reading data from a csv.
        df.loc[0, "get_station_kwargs"] = str(df.loc[0, "get_station_kwargs"])
        # now add a row with an empty get_station_kwargs column
        old = dict(df.iloc[0])
        new = {
            "station": "CWU",
            "network": "UU",
            "channel": "EHZ",
            "location": "01",
            "seed_id": "UU.CWU.01.EHZ",
            "get_station_kwargs": "{}",
        }
        old.update(new)
        ser = pd.Series(old)
        return pd.concat([df, ser.to_frame().T], axis=0).reset_index(drop=True)

    @pytest.fixture
    def df_with_partial_responses(self, df_with_nrl_response):
        """test creating inv with partial responses."""
        # set one row to None
        df_with_nrl_response.loc[0, "sensor_keys"] = None
        return df_with_nrl_response

    @pytest.fixture
    def df_with_both_response_cols(
        self, df_with_nrl_response, df_with_get_stations_kwargs
    ):
        """
        Get a df with both types of responses.

        row 1 should have NRL, row 2 both, and row three client kwargs
        """
        df1, df2 = df_with_nrl_response, df_with_get_stations_kwargs
        df1["get_station_kwargs"] = df2["get_station_kwargs"]
        df1.loc[2, ("sensor_keys", "datalogger_keys")] = (None, "")
        return df1

    @pytest.fixture
    def df_ambiguous_client_query(self, df_with_get_stations_kwargs):
        """
        Get a dataframe which has get_station_kwargs which will pull more
        than one station from IRIS.
        """
        df = df_with_get_stations_kwargs.copy()
        df.loc[1, "get_station_kwargs"]["channel"] = "*"
        return df

    def test_nrl_responses(self, df_with_nrl_response):
        """Ensure the NRL is used to pull responses."""
        with suppress_warnings():
            inv = df_to_inventory(df_with_nrl_response)
        assert self.has_valid_response(inv)

    def test_response_one_missing(self, df_with_partial_responses):
        """Ensure responses which can be got are fetched."""
        df = df_with_partial_responses
        with suppress_warnings():
            inv = df_to_inventory(df)

        missing = df["sensor_keys"].isnull() | df["datalogger_keys"].isnull()
        missing_seed_ids = set(get_seed_id_series(df[missing]))
        assert self.has_valid_response(inv, missing_seed_ids)

    def test_get_stations_client(self, df_with_get_stations_kwargs):
        """Ensure get_station_kwargs results responses."""
        df = df_with_get_stations_kwargs
        col = "get_station_kwargs"
        missing = df[col].isnull() | (df[col] == "")
        missing_seed_ids = set(get_seed_id_series(df[missing]))

        with suppress_warnings():
            inv = df_to_inventory(df)

        assert self.has_valid_response(inv, missing_seed_ids)

    def test_mixing_nrl_with_station_client(self, df_with_both_response_cols):
        """
        Ensure mixing the two types of responses raises an Error.
        """
        df = df_with_both_response_cols
        with pytest.raises(AmbiguousResponseError):
            df_to_inventory(df)

    def test_ambiguous_query_raises(self, df_ambiguous_client_query):
        """Ensure a query that returns multiple channels will raise."""

        df = df_ambiguous_client_query
        with pytest.raises(AmbiguousResponseError):
            df_to_inventory(df)


class TestGetStationClient:
    """Tests for getting a station client from various inputs."""

    def test_inventory(self, bingham_dataset):
        """Ensure an inventory returns an inventory."""
        client = get_station_client(bingham_dataset.station_client)
        assert isinstance(client, StationClient)
        assert isinstance(client, obspy.Inventory)

    def test_inventory_on_disk(self, bingham_dataset):
        """Ensure a path to an obspy-readable inventory works."""
        path = bingham_dataset.station_path / "UU.NOQ.xml"
        client = get_station_client(path)
        assert isinstance(client, StationClient)
        assert isinstance(client, obspy.Inventory)

    def test_not_a_client(self):
        """ensure a non station-client-able object raises."""
        with pytest.raises(TypeError):
            get_station_client(1)
