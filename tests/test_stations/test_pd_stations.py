"""
Tests for gettting station dataframes from objects.
"""
import os
import tempfile
from os.path import join, exists

import numpy as np
import obspy
import pandas as pd
import pytest

import obsplus
from obsplus import stations_to_df
from obsplus.constants import STATION_COLUMNS
from obsplus.utils.time import is_time

STA_COLUMNS = {"latitude", "longitude", "elevation", "start_date", "end_date"}


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
        pass


class TestReadInventory:
    """ ensure inventories can be read in """

    fixtures = ["df_from_inv", "df_from_inv_df"]

    # fixtures
    @pytest.fixture(scope="class")
    def df_from_inv(self):
        """ read events from a events object """
        inv = obspy.read_inventory()
        return stations_to_df(inv)

    @pytest.fixture(scope="class")
    def df_from_inv_df(self):
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
        f1 = tmpdir.mkdir("data").join("stations.csv")
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
        """ ensure numeric network, station, location, codes are interpreted
        as strs """
        df = stations_to_df(str(numeric_csv))
        for col in ["network", "station", "location", "channel"]:
            ser = df[col]
            for val in ser.values:
                assert isinstance(val, str)


class TestReadDirectoryOfInventories:
    """ ensure that a directory of various stations files can be read """

    nest_name = "nest"

    # help functions
    def nest_directly(self, nested_times, path):
        """ make a directory nested n times """
        nd_name = join(path, self.nest_name)
        if not exists(nd_name) and nested_times:
            os.makedirs(nd_name)
        elif not nested_times:  # recursion limit reached
            return path
        return self.nest_directly(nested_times - 1, nd_name)

    # fixtures
    @pytest.fixture(scope="class")
    def inventory(self, kemmerer_dataset):
        """ read the stations """
        return kemmerer_dataset.station_client.get_stations()

    @pytest.fixture(scope="class")
    def inv_directory(self, inventory):
        """ create a nested directory of inventories """
        chans = inventory.get_contents()["channels"]
        # make a silly nested directory
        with tempfile.TemporaryDirectory() as tempdir:
            for num, seed_id in enumerate(chans):
                network, station, location, channel = seed_id.split(".")
                inv = inventory.select(channel=channel, station=station)
                file_name = seed_id + ".xml"
                write_path = join(self.nest_directly(num, tempdir), file_name)
                inv.write(write_path, "stationxml")
            yield tempdir

    @pytest.fixture(scope="class")
    def read_inventory(self, inv_directory):
        return stations_to_df(inv_directory)

    # tests
    def test_something(self, read_inventory, inventory):
        inv_df = stations_to_df(inventory)
        assert (read_inventory.columns == inv_df.columns).all()
        assert not read_inventory.empty
        assert len(inv_df) == len(read_inventory)
        assert set(inv_df["seed_id"]) == set(read_inventory["seed_id"])


class TestReadKemInventory:
    """ read the kemmerer inventories (csv and xml) and run tests """

    kem_ds = obsplus.load_dataset("kemmerer")
    csv_path = kem_ds.source_path / "inventory.csv"
    sml_path = kem_ds.source_path / "inventory.xml"
    sml = obspy.read_inventory(str(sml_path))
    df = pd.read_csv(csv_path)
    supported_inputs = ["sml_path", "sml", "csv_path", "df"]

    # fixtures
    @pytest.fixture(scope="class", params=supported_inputs)
    def inv_df(self, request):
        """ collect all the supported inputs are parametrize"""
        name = request.param
        return stations_to_df(getattr(self, name))

    # tests
    def test_size(self, inv_df):
        """ ensure the correct number of items is in df """
        assert len(inv_df) == len(self.df)

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
        df = stations_to_df(station_cache_inventory)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_kem_catalog(self):
        """ test converting the kemmerer catalog to an inv dataframe. """
        ds = obsplus.load_dataset("kemmerer")
        df = stations_to_df(ds.event_client.get_events())
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
