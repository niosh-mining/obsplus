"""
tests for get events
"""
import copy

import obspy
import pandas as pd
import pytest
from obsplus.utils.misc import suppress_warnings
from obspy.core.event import CreationInfo
from obspy.geodetics import gps2dist_azimuth


from obsplus.events.get_events import get_events

# ---------------------- module fixtures


@pytest.fixture
def catalog():
    """ return a copy of the default events """
    return obspy.read_events()


# --------------------- tests


class TestGetEvents:
    """ tests the get events interface of the events object """

    # tests
    def test_catalog_monkey_patched(self, catalog):
        """ ensure the get_events method was monkey patched to events """
        assert hasattr(catalog, "get_events")

    def test_filter_doesnt_modify_original(self, catalog):
        """ ensure calling get_events doesn't modify original """
        cat_before = copy.deepcopy(catalog)
        catalog.get_events()
        assert cat_before == catalog

    def test_return_type(self, catalog):
        """ ensure a events is returned """
        cat = catalog.get_events()
        assert isinstance(cat, obspy.Catalog)

    def test_unsupported_param_raises(self, catalog):
        """ ensure query params that are not supported raise error """
        from obsplus.events.get_events import UNSUPPORTED_PARAMS

        for bad_param in UNSUPPORTED_PARAMS:
            with pytest.raises(TypeError):
                catalog.get_events(**{bad_param: 1})

    def test_filter_lat_lon(self, catalog):
        """ test that the events can be filtered """
        cat_out = catalog.get_events(maxlatitude=39, maxlongitude=41)
        assert len(cat_out) == 1

    def test_filter_event_id(self, catalog):
        """ test that ids can be used to filter """
        eveid = catalog[0].resource_id
        out = catalog.get_events(eventid=eveid)
        assert len(out) == 1
        assert out[0] == catalog[0]

    def test_update_after(self, catalog):
        """ test that ids can be used to filter """
        eve = catalog[0]
        time = obspy.UTCDateTime("2017-05-04")
        eve.creation_info = CreationInfo(creation_time=time)
        out = catalog.get_events(updatedafter=time - 2)
        assert len(out) == 1
        assert out[0] == catalog[0]

    def test_none_doesnt_effect_output(self, catalog):
        """ ensure parameters passed None dont get applied as filters """
        cat1 = catalog.get_events()
        cat2 = catalog.get_events(minlatitude=None, maxlatitude=None)
        assert len(cat1) == len(cat2)
        assert cat1 == cat2

    def test_radius_degrees(self, catalog):
        """ ensure the max_radius works with degrees specified. """
        lat, lon = 39.342, 41.044
        with suppress_warnings():
            cat = catalog.get_events(latitude=lat, longitude=lon, maxradius=8)
        # For the example catalog there should be exactly 2 events included
        assert len(cat) == 2

    def test_circular_search_no_events(self, catalog):
        """ Ensure no errors if no events within rectangular region. #177 """
        lat, lon = 31.0, 30.0
        with suppress_warnings():
            cat = catalog.get_events(latitude=lat, longitude=lon, maxradius=1)
        assert len(cat) == 0

    def test_max_min_radius_m(self, catalog):
        """ Ensure max and min radius work in m (ie when degrees=False). """
        minrad = 10000
        maxrad = 800_000
        lat, lon = 39.342, 41.044
        kwargs = dict(
            latitude=lat,
            longitude=lon,
            minradius=minrad,
            maxradius=maxrad,
            degrees=False,
        )

        with suppress_warnings():  # suppress geograhiclib warning
            df = catalog.get_events(**kwargs).get_event_summary()

        for _, row in df.iterrows():
            args = (row.latitude, row.longitude, lat, lon)
            dist, _, _ = gps2dist_azimuth(*args)
            assert minrad < dist < maxrad


class TestGetEventSummary:
    """ tests for returning and event summary dataframe """

    def test_is_dataframe(self, catalog):
        """ ensure an non-empty events is returned """
        df = catalog.get_event_summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df)


class TestDateLineQuery:
    """
    Tests for getting events spanning the dateline.
    """

    def test_get_event_query_around_earth(self, dateline_catalog):
        """Ensure getting events can reach across dateline."""
        kwargs = dict(minlongitude=-179, maxlongitude=179)
        out = get_events(dateline_catalog, **kwargs)
        assert len(out) == 1
        assert out[0].origins[0].longitude == 0
        assert out[0].origins[0].latitude == 0

    def test_get_event_query_over_dateline(self, dateline_catalog):
        """Test for just querying over the dateline."""
        kwargs = dict(minlongitude=179, maxlongitude=-179)
        out = get_events(dateline_catalog, **kwargs)
        # should return both events
        assert len(out) == 2
