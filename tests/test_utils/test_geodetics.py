"""
Tests for geodetic utils.
"""
import itertools

import numpy as np
import obspy
import pandas as pd
import pytest

import obsplus
from obsplus.constants import DISTANCE_COLUMN_DTYPES
from obsplus.utils.geodetics import SpatialCalculator, map_longitudes
from obsplus.exceptions import DataFrameContentError
from obsplus.utils.misc import suppress_warnings


class TestCalculateDistance:
    """
    Tests for calculating distance from a single location to a dataframe.
    """

    @pytest.fixture(scope="class")
    def spatial_calc(self):
        """ return the default instance of the spatial calculator. """
        return SpatialCalculator()

    @pytest.fixture(scope="class")
    def cat(self):
        """ return the first 3 events from the crandall_test dataset. """
        return obspy.read_events()

    @pytest.fixture(scope="class")
    def inv(self):
        """Return the default inventory."""
        return obspy.read_inventory()

    @pytest.fixture(scope="class")
    def distance_df(self, cat, inv, spatial_calc):
        """ Return a dataframe from all the crandall_test events and stations. """
        with suppress_warnings():
            return spatial_calc(entity_1=cat, entity_2=inv)

    def test_type(self, distance_df):
        """ ensure a dataframe was returned. """
        assert isinstance(distance_df, pd.DataFrame)
        assert set(distance_df.columns) == set(DISTANCE_COLUMN_DTYPES)

    def test_all_events_in_df(self, distance_df, cat):
        """ Ensure all the events are in the distance dataframe. """
        event_ids_df = set(distance_df.index.to_frame()["id1"])
        event_ids_cat = {str(x.resource_id) for x in cat}
        assert event_ids_cat == event_ids_df

    def test_all_seed_id_in_df(self, distance_df, inv):
        """Ensure all the seed ids are in the dataframe."""
        seed_id_stations = set(obsplus.stations_to_df(inv)["seed_id"])
        seed_id_df = set(distance_df.index.to_frame()["id2"])
        assert seed_id_df == seed_id_stations

    def test_cat_cat(self, cat, spatial_calc):
        """ ensure it works with two catalogs """
        with suppress_warnings():
            df = spatial_calc(cat, cat)
        event_ids = {str(x.resource_id) for x in cat}
        combinations = set(itertools.product(event_ids, event_ids))
        assert combinations == set(df.index)

    def test_dataframe_input(self, cat, spatial_calc):
        """
        Any dataframe should be valid input provided it has the required columns.
        """
        data = [[10.1, 10.1, 0, "some_id"]]
        cols = ["latitude", "longitude", "elevation", "id"]
        df = pd.DataFrame(data, columns=cols).set_index("id")
        with suppress_warnings():
            dist_df = spatial_calc(df, cat)
        # make sure output has expected shape and ids
        assert len(dist_df) == len(cat) * len(df)
        id1 = dist_df.index.get_level_values("id1")
        assert "some_id" in set(id1)

    def test_with_tuple_no_id(self, spatial_calc):
        """ Test getting relations with tuple and catalog. """
        cat = obspy.read_events()
        df = obsplus.events_to_df(cat)
        ser = df.iloc[0]
        # first test with no id
        tuple1 = (ser["latitude"], ser["longitude"], -ser["depth"])
        with suppress_warnings():
            out1 = spatial_calc(cat, tuple1)
        # the default index should be sequential
        assert set(out1.index.get_level_values("id2")) == {0}
        # expected len is 3
        assert len(out1) == 3

    def test_tuple_with_id(self, spatial_calc):
        """ Ensure if a 4th column is given the id works. """
        tuple1 = (45, -111, 0, "bob")
        tuple2 = (45, -111, 0)
        with suppress_warnings():
            out = spatial_calc(tuple1, tuple2)
        # distances should be close to 0
        dist_cols = ["distance_m", "distance_degrees", "vertical_distance_m"]
        distances = out[dist_cols].values
        assert np.allclose(distances, 0)
        # azimuths should be increments of 180
        assert np.allclose(out["azimuth"] % 180, 0)
        assert np.allclose(out["back_azimuth"] % 180, 0)
        # check length and index names
        assert len(out) == 1
        assert set(out.index.to_list()) == {("bob", 0)}

    def test_short_sequence(self, spatial_calc):
        """ A sequence which is too short should raise. """
        input1 = [(45, -111, 0), (46, -111, 0), (49, -112, 0)]
        input2 = (45, -111)
        with pytest.raises(ValueError):
            spatial_calc(input1, input2)

    def test_list_of_tuples(self, spatial_calc):
        """Test a list of tuples."""
        input1 = [(45, -111, 0), (46, -111, 0), (49, -112, 0)]
        input2 = obspy.read_events()
        with suppress_warnings():
            out = spatial_calc(input1, input2)
        assert len(out) == 9
        assert not out.isnull().any().any()

    def test_duplicated_id_different_coords_raises(self, spatial_calc):
        """ Duplicated indices and with different coords should raise. """
        cat = obspy.read_events() + obspy.read_events()
        # change preferred origin on a duplicated id
        origin = cat[0].preferred_origin()
        origin.latitude = origin.latitude / 2.0
        with pytest.raises(ValueError) as e:
            spatial_calc(cat, (45, -111, 0))
        assert "multiple coordinates for" in e.value.args[0]

    def test_invalid_df(self, spatial_calc, cat, inv):
        """ Ensure dfs with missing columns raise. """
        df1 = cat.to_df().drop(columns="latitude")
        df2 = inv.to_df().drop(columns="latitude")
        with pytest.raises(
            DataFrameContentError,
            match="SpatialCalculator input dataframe must have the following",
        ):
            spatial_calc(df1, inv)
        with pytest.raises(
            DataFrameContentError,
            match="SpatialCalculator input dataframe must have the following",
        ):
            spatial_calc(cat, df2)

    def test_invalid_lat_lon(self, spatial_calc, cat, inv):
        """ Ensure invalid latitudes or longitudes get flagged """
        df = inv.to_df()
        df["latitude"] = 200
        with pytest.raises(DataFrameContentError, match="invalid lat/lon"):
            spatial_calc(cat, df)


class TestMapLongitudes:
    """
    Tests for mapping angles to a specific domain.
    """

    def test_in_domain(self):
        """Ensure angles already in the domain don't get mapped."""
        ar = np.array([10, 100, -70])
        out = map_longitudes(ar)
        assert np.allclose(ar, out)

    def test_mapped_angles(self):
        """Ensure angles outside the domain get mapped into it."""
        ar = np.array([190, 271, 361, -719.25, 10, -10])
        expected = np.array([-170, -89, 1, 0.75, 10, -10])
        assert np.allclose(map_longitudes(ar), expected)
