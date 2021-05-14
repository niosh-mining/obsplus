"""
Tests for gettting station dataframes from objects.
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
import pytest

import obsplus
from obsplus import stations_to_df
from obsplus.constants import STATION_COLUMNS, pd_time_types
from obsplus.utils.misc import register_func, suppress_warnings

STA_COLUMNS = {"latitude", "longitude", "elevation", "start_date", "end_date"}


def is_time(obj):
    """Return True if an object is a time type."""
    return isinstance(obj, pd_time_types) or pd.isnull(obj)


class TestInv2Df:
    """ tests for the stations to dataframe method """

    # fixtures
    @pytest.fixture(scope="class")
    def invdf(self, test_inventory):
        """ return the dataframe produced from stations"""
        return stations_to_df(test_inventory)

    # tests
    def test_method_exits(self, test_inventory):
        """ make sure stations has the catalog2df method """
        assert hasattr(test_inventory, "to_df")

    def test_output(self, invdf):
        """ Simple check on station outputs. """
        assert isinstance(invdf, pd.DataFrame)
        assert len(invdf)
        assert STA_COLUMNS.issubset(invdf.columns)
        for ind, row in invdf.iterrows():
            t1, t2 = row.start_date, row.end_date
            assert is_time(t1) and is_time(t2)
            assert isinstance(row.seed_id, str)

    def test_to_df_method(self):
        """ ensure the to_df method is attached and works. """
        inv = obspy.read_inventory()
        chans = inv.get_contents()["channels"]
        df = inv.to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(chans) == len(df)

    def test_time_columns(self, invdf):
        """ ensure the times are np.datetime instances. """
        assert invdf["start_date"].dt  # if not dt this will raise
        assert invdf["end_date"].dt

    @pytest.mark.parametrize(
        "input,expected", [(1, "01"), (1.0, "01"), ("01", "01"), ("1", "1")]
    )
    def test_location_codes(self, invdf, input, expected):
        """ make sure location codes are handled nicely """
        invdf = invdf.copy()
        if isinstance(input, int):
            invdf = invdf.loc[invdf["station"] == "RJOB"]
            invdf["location"] = input
            invdf["location"] = invdf["location"].astype(int)
        elif isinstance(input, float):
            invdf["location"] = np.nan
            invdf.loc[invdf["station"] == "RJOB", "location"] = input
        else:
            invdf.loc[invdf["station"] == "RJOB", "location"] = input
        # breakpoint()
        invdf = stations_to_df(invdf)
        rjob = invdf.loc[invdf["station"] == "RJOB"]
        not_rjob = invdf.loc[invdf["station"] != "RJOB"]
        assert (rjob["location"] == expected).all()
        assert (not_rjob["location"] == "").all()
        assert (
            invdf["location"] == invdf["seed_id"].str.split(".", expand=True)[2]
        ).all()  # This one will be a little bit tougher


class TestReadInventory:
    """ ensure inventories can be read in """

    fixtures = []

    # fixtures
    @pytest.fixture(scope="class")
    @register_func(fixtures)
    def df_from_inv(self):
        """ read events from a events object """
        inv = obspy.read_inventory()
        return stations_to_df(inv)

    @pytest.fixture(scope="class")
    @register_func(fixtures)
    def df_from_inv_df(self):
        """Return a df from an inventory dataframe."""
        event_dict = {
            "start_date": obspy.UTCDateTime(),
            "end_date": None,
            "latitude": 41,
            "longitude": -111.1,
            "elevation": 1.0,
            "network": "LF",
            "station": "TEX",
            "location": None,
            "channel": "HHZ",
            "seed_id": "LF.TEX.HHZ",
        }
        df = pd.DataFrame(pd.Series(event_dict)).T
        return stations_to_df(df)

    @pytest.fixture(scope="class", params=fixtures)
    def read_inventory_output(self, request):
        """ the parametrized output of read_events fixtures """
        return request.getfixturevalue(request.param)

    @pytest.fixture
    def numeric_csv(self, tmpdir):
        """ write a csv with numeric net/sta/loc codes, return path """
        f1 = Path(tmpdir.mkdir("data")) / "stations.csv"
        t1 = obspy.UTCDateTime("2012-01-01").timestamp
        t2 = t1 + 3600
        data = [
            ("01", "01", "01", "01", "", 100, 50, 100, t1, t2),
            ("01", "02", "01", "01", "", 100, 50, 100, t1, t2),
            ("01", "03", "01", "01", "", 100, 50, 100, t1, t2),
        ]
        cols = [
            "network",
            "station",
            "location",
            "channel",
            "seed_id",
            "latitude",
            "longitude",
            "elevation",
            "start_date",
            "end_date",
        ]
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(f1, index=False)
        return f1

    # tests
    def test_basics(self, read_inventory_output):
        """ make sure a dataframe is returned """
        assert isinstance(read_inventory_output, pd.DataFrame)
        assert len(read_inventory_output)

    def test_read_inv_with_numeric_codes(self, numeric_csv):
        """
        Ensure numeric network, station, location, codes are interpreted
        as strs
        """
        df = stations_to_df(str(numeric_csv))
        for col in ["network", "station", "location", "channel"]:
            ser = df[col]
            for val in ser.values:
                assert isinstance(val, str)

    def test_gather(self, df_from_inv, df_from_inv_df):
        """ Simply gather aggregated fixtures so they are marked as used. """


class TestReadDirectoryOfInventories:
    """ ensure that a directory of various stations files can be read """

    nest_name = "nest"

    # help functions
    def nest_directly(self, nested_times, path):
        """ make a directory nested n times """
        nd_name = Path(path) / self.nest_name
        if not Path(nd_name).exists() and nested_times:
            os.makedirs(nd_name)
        elif not nested_times:  # recursion limit reached
            return path
        return self.nest_directly(nested_times - 1, nd_name)

    # fixtures
    @pytest.fixture()
    def inventory(self, bingham_dataset):
        """ read the stations """
        # get sub-inventory
        client = bingham_dataset.station_client
        stations = ["COY", "FFT", "RIV"]
        inv1 = client.get_stations(station=stations[0])
        for station in stations[1:]:
            inv1 += client.get_stations(station=station)
        return inv1

    @pytest.fixture()
    def inv_directory(self, inventory):
        """ create a nested directory of inventories """
        chans = inventory.get_contents()["channels"]
        # make a silly nested directory
        with tempfile.TemporaryDirectory() as tempdir:
            for num, seed_id in enumerate(chans):
                network, station, location, channel = seed_id.split(".")
                inv = inventory.select(channel=channel, station=station)
                file_name = seed_id + ".xml"
                nest_dir = self.nest_directly(num, tempdir)
                write_path = Path(nest_dir) / file_name
                inv.write(str(write_path), "stationxml")
            yield tempdir

    @pytest.fixture()
    def read_inventory(self, inv_directory):
        """Convert the inventory directory to a dataframe."""
        with suppress_warnings():
            return stations_to_df(inv_directory)

    # tests
    def test_read_inventory_directories(self, read_inventory, inventory):
        """Tests for reading a directory of inventories."""
        with suppress_warnings():
            inv_df = stations_to_df(inventory)
        assert (read_inventory.columns == inv_df.columns).all()
        assert not read_inventory.empty
        assert len(inv_df) == len(read_inventory)
        assert set(inv_df["seed_id"]) == set(read_inventory["seed_id"])


class TestReadTAInventory:
    """ read the ta_test inventories (csv and xml) and run tests """

    fixtures = []

    @pytest.fixture(scope="class")
    @register_func(fixtures)
    def ta_inventory(self, ta_dataset):
        """ Return the ta_test inventory """
        return ta_dataset.station_client.get_stations()

    @pytest.fixture(scope="class")
    @register_func(fixtures)
    def ta_inv_df(self, ta_inventory):
        """ Return the ta_test inventory as a dataframe. """
        return obsplus.stations_to_df(ta_inventory)

    @pytest.fixture(scope="class")
    @register_func(fixtures)
    def inventory_csv_path(self, ta_inv_df, tmp_path_factory):
        """ Return a csv path to the ta_test inventory. """
        path = Path(tmp_path_factory.mktemp("tempinvta")) / "inv.csv"
        ta_inv_df.to_csv(path, index=False)
        return path

    @pytest.fixture(scope="class")
    @register_func(fixtures)
    def inventory_xml_path(self, ta_inventory, tmp_path_factory):
        """ Return a to the ta_test inventory saved as an xml. """
        path = Path(tmp_path_factory.mktemp("tempinvta")) / "inv.xml"
        ta_inventory.write(str(path), "stationxml")
        return path

    @pytest.fixture(scope="class", params=fixtures)
    def inv_df(self, request):
        """ collect all the supported inputs are parametrize"""
        value = request.getfixturevalue(request.param)
        return stations_to_df(value)

    # tests
    def test_size(self, inv_df, ta_inventory):
        """ ensure the correct number of items is in df """
        channel_count = len(ta_inventory.get_contents()["channels"])
        assert len(inv_df) == channel_count

    def test_column_order(self, inv_df):
        """ ensure the order of the columns is correct """
        cols = list(inv_df.columns)
        assert list(STATION_COLUMNS) == cols[: len(STATION_COLUMNS)]

    def test_datetime_columns(self, inv_df):
        """ start_date and end_date should be UTCDateTime objects """
        assert all([is_time(x) for x in inv_df["start_date"]])
        assert all([is_time(x) for x in inv_df["end_date"]])


class TestReadDataFrame:
    """ tests for reading dataframes """

    # fixtures
    @pytest.fixture
    def inv_df(self):
        """ return a small dataframe for manipulating """
        df = stations_to_df(obspy.read_inventory())
        return df

    @pytest.fixture
    def df_bad_location(self, inv_df):
        """ make location codes nan, run through read_inventory """
        inv_df.loc[:, "location"] = np.NaN
        return stations_to_df(inv_df)

    # tests
    def test_idempotency(self, inv_df):
        """ ensure the inv_df function is idempotent """
        inv_df2 = stations_to_df(inv_df)
        assert inv_df2.equals(inv_df)
        assert inv_df2 is not inv_df

    def test_bad_locations_handled(self, df_bad_location):
        """ ensure the NaN location codes are changed to blank str """
        assert (df_bad_location.loc[:, "location"] == "").all()


class TestStationDfFromCatalog:
    """ test to read stations like data from catalogs/events """

    # tests
    def test_basic_inventories(self, station_cache_inventory):
        """Test for all basic inventories."""
        df = stations_to_df(station_cache_inventory)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_kem_catalog(self, bingham_dataset):
        """ test converting the kemmerer catalog to an inv dataframe. """
        df = stations_to_df(bingham_dataset.event_client.get_events())
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


class TestStationDfFromWaveBank:
    """ Test that stations info can be extracted from the wavebank. """

    @pytest.fixture(scope="class")
    def wavebank_station_df(self, crandall_bank):
        """ Return the station df from a wavebank """
        return stations_to_df(crandall_bank)

    def test_df_returned(self, wavebank_station_df):
        """ a df should be returned and not empty. """
        assert isinstance(wavebank_station_df, pd.DataFrame)
        assert len(wavebank_station_df)


class TestStationDfFromStream:
    """Ensure station data can be extracted from a stream."""

    def test_stream_to_inv(self):
        """A stream also contains station info."""
        st = obspy.read()
        df = obsplus.stations_to_df(st)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(st)
