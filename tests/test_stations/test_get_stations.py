"""
tests for get stations
"""
from pathlib import Path

import obspy
import pytest

import obsplus
import pandas as pd
import obsplus.utils.pd
from obsplus.constants import NSLC
from obsplus.utils.stations import df_to_inventory
from obsplus.utils.stations import get_station_client


@pytest.fixture
def inventory():
    """Return the default inventory."""
    return obspy.read_inventory()


class TestGetStation:
    """Test for getting stations from inventory."""

    @pytest.fixture
    def inv_issue_115(self):
        """Get an inventory for testing issue 115."""

        sta1 = dict(
            network="LF",
            location="",
            station="BOB",
            channel="HHZ",
            start_date="2019-01-01",
            end_date="2100-01-01",
            sample_rate=250,
            latitude=0,
            longitude=0,
            elevation=0,
            depth=0,
        )
        sta2 = dict(
            network="01",
            location="",
            station="01",
            channel="BHZ",
            start_date="2019-01-01",
            end_date="2100-01-01",
            sample_rate=1000,
            latitude=0,
            longitude=0,
            elevation=0,
            depth=0,
        )

        return df_to_inventory(pd.DataFrame([sta1, sta2]))

    def test_inv_has_get_stations(self, inventory):
        """get stations should have been monkey patched to stations"""
        assert hasattr(inventory, "get_stations")

    def test_return_type(self, inventory):
        """ensure an stations type is returned"""
        assert isinstance(inventory.get_stations(), obspy.Inventory)

    def test_filter_on_lat_lon(self, inventory):
        """ensure stations can be filtered on lat/lon"""
        lat = 48.162899
        lon = 11.275200
        kwargs = dict(
            minlatitude=lat - 0.01,
            maxlatitude=lat + 0.01,
            minlongitude=lon - 0.01,
            maxlongitude=lon + 0.01,
        )
        inv = inventory.get_stations(**kwargs)
        df = obsplus.stations_to_df(inv)
        assert set(df.station) == {"FUR"}

    def test_filter_station(self, inventory):
        """ensure stations can be filtered"""
        inv = inventory.get_stations(station="WET")
        df = obsplus.stations_to_df(inv)
        assert set(df.station) == {"WET"}

    def test_filter_channel_single_wild(self, inventory):
        """ensure filtering can be done on str attrs with ?"""
        inv = inventory.get_stations(channel="HH?")
        df = obsplus.stations_to_df(inv)
        assert all([x.startswith("HH") for x in set(df.channel)])

    def test_filter_channel_star_wild(self, inventory):
        """ensure filtering can be done with *"""
        inv = inventory.get_stations(channel="*z")
        df = obsplus.stations_to_df(inv)
        assert all([x.endswith("Z") for x in set(df.channel)])

    def test_get_stations_one_channel(self, inventory):
        """test get stations when all kwarg are used."""
        sta_df = obsplus.stations_to_df(inventory)
        nslc = obsplus.utils.pd.get_seed_id_series(sta_df).iloc[0]
        # make kwargs
        kwargs = {x: y for x, y in zip(NSLC, nslc.split("."))}
        # get sub inv
        inv = inventory.get_stations(**kwargs)
        df_out = obsplus.stations_to_df(inv)
        assert len(df_out) == 1
        # iterate through each net, sta, chan, etc. and check length
        assert len(inv.networks) == 1
        for net in inv:
            assert len(net.stations) == 1
            for sta in net:
                assert len(sta.channels) == 1

    def test_get_stations_issue_115(self, inv_issue_115):
        """Test that issue 115 is fixed."""
        # first filter on channel, then ensure the channels and stations
        # without that channel get removed
        out = inv_issue_115.get_stations(channel="HHZ")
        assert len(out) == 1, "there should now be one network"
        assert len(out[0]) == 1, "and one station"

    def test_read_inventory_from_file(self, tmpdir):
        """Should be able to read file into inventory."""
        path = Path(tmpdir) / "inv"
        inv = obspy.read_inventory()
        inv.write(str(path), "stationxml")
        assert inv == get_station_client(str(path))
