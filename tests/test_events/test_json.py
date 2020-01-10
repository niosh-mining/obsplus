"""
tests for converting to/from json
"""

import tempfile

import obspy
import obspy.core.event as ev
import pytest
from obspy.core.event import CreationInfo, Catalog, Event

from obsplus import json_to_cat, cat_to_dict, cat_to_json
from obsplus.utils.misc import yield_obj_parent_attr


def _remove_empty_quantity_errors(catalog):
    """
    Copy the catalog and set all empty QunatityErrors to None.
    This is needed to evaluate equality of catalogs that may have
    None or empty QuantityErrors.

    Fixed in https://github.com/obspy/obspy/pull/2185
    """
    cat = catalog.copy()
    for obj, parent, attr in yield_obj_parent_attr(cat, cls=ev.QuantityError):
        if not obj:
            setattr(parent, attr, None)
    return cat


class TestCat2Json:
    """ class to test json """

    # fixtures
    @pytest.fixture(scope="class")
    def json_from_cat(self, test_catalog):
        """ Call the to_json method to return json string """
        return cat_to_json(test_catalog)

    @pytest.fixture(scope="class")
    def cat_from_json(self, json_from_cat):
        """ load the json into a cat_name object """
        cat = json_to_cat(json_from_cat)
        return cat

    @pytest.fixture(scope="class")
    def json_cat_from_disk(self, cat_from_json):
        """ save the json events to disk and read it again into memory """
        tf = tempfile.mkstemp()
        cat_from_json = obspy.Catalog(cat_from_json)
        cat_from_json.write(tf[1], "quakeml")
        cat = obspy.read_events(tf[1])
        return cat

    # tests
    def test_to_json(self, json_from_cat):
        """ test that the returned json is a string """
        assert isinstance(json_from_cat, str)

    def test_load_json(self, cat_from_json, test_catalog):
        """ test that the json can be loaded into a cat_name """
        # ensure a events was returned
        assert isinstance(cat_from_json, obspy.Catalog)
        # catalogs should be equal after accounting for QunatityErrors
        cat1 = _remove_empty_quantity_errors(cat_from_json)
        cat2 = _remove_empty_quantity_errors(test_catalog)
        assert cat1 == cat2

    def test_catalog_can_be_written(self, test_catalog, json_cat_from_disk):
        """ ensure the events can be written then read in again and is
        still equal """
        assert test_catalog == json_cat_from_disk


class TestSerializeUTCDateTime:
    """ ensure obsplus can serialize datetimes without loss.
    motivated by https://github.com/obspy/obspy/issues/2034 """

    # timestamps to test that can be serialized
    times = [
        1515174511.1984465,
        1515174511.1984463,
        1515174511.1984460,
        515174511.1984458,
        1515174511.1984459,
        0.1984465,
        0.1984463,
    ]

    # helper functions
    @staticmethod
    def create_catalog(time):
        """ create a events object with a UTCTimeObject as event creation
        info """
        creation_info = CreationInfo(creation_time=obspy.UTCDateTime(time))
        event = Event(creation_info=creation_info)
        return Catalog(events=[event])

    # fixtures
    @pytest.fixture(params=times)
    def cat(self, request):
        """ create a events using value from time in the creation info """
        return self.create_catalog(request.param)

    @pytest.fixture
    def cat2(self, cat):
        """ serialize events, then load in in again and return """
        json = cat_to_dict(cat)
        return json_to_cat(json)

    # tests
    def test_equal(self, cat, cat2):
        """ ensure the catalogs are equal before and after serialization """
        assert cat == cat2
