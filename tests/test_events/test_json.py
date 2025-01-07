"""
tests for converting to/from json
"""

import tempfile
from typing import ClassVar

import obspy
import obspy.core.event as ev
import pytest
from obsplus import cat_to_dict, cat_to_json, json_to_cat
from obsplus.utils.misc import yield_obj_parent_attr
from obspy.core.event import Catalog, CreationInfo, Event


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
    """class to test json"""

    # fixtures
    @pytest.fixture(scope="class")
    def json_from_cat(self, test_catalog):
        """Call the to_json method to return json string"""
        return cat_to_json(test_catalog)

    @pytest.fixture(scope="class")
    def cat_from_json(self, json_from_cat):
        """Load the json into a cat_name object"""
        cat = json_to_cat(json_from_cat)
        return cat

    @pytest.fixture(scope="class")
    def json_cat_from_disk(self, cat_from_json):
        """Save the json events to disk and read it again into memory"""
        tf = tempfile.mkstemp()
        new_cat_from_json = cat_from_json
        new_cat_from_json.write(tf[1], "quakeml")
        cat = obspy.read_events(tf[1])
        return cat

    # tests
    def test_to_json(self, json_from_cat):
        """Test that the returned json is a string"""
        assert isinstance(json_from_cat, str)

    def test_load_json(self, cat_from_json, test_catalog):
        """Test that the json can be loaded into a cat_name"""
        # ensure a events was returned
        assert isinstance(cat_from_json, obspy.Catalog)
        # catalogs should be equal after accounting for QunatityErrors
        cat1 = _remove_empty_quantity_errors(cat_from_json)
        cat2 = _remove_empty_quantity_errors(test_catalog)
        assert cat1 == cat2

    def test_catalog_can_be_written(self, test_catalog, json_cat_from_disk):
        """
        Ensure the events can be written then read in again and is
        still equal.
        """
        assert test_catalog == json_cat_from_disk

    def test_single_event(self):
        """Ensure a single event can be converted to json."""
        event = obspy.read_events()[0]
        out = cat_to_json(event)
        assert isinstance(out, str)
        cat_again = json_to_cat(out)
        assert len(cat_again) == 1

    def test_list_of_events(self):
        """Ensure a list of events can be converted to json."""
        events = obspy.read_events().events
        out = cat_to_json(events)
        assert isinstance(out, str)
        cat_again = json_to_cat(out)
        assert len(cat_again) == len(events)


class TestSerializeUTCDateTime:
    """
    Ensure obsplus can serialize datetimes without loss.
    motivated by https://github.com/obspy/obspy/issues/2034
    """

    # timestamps to test that can be serialized
    times: ClassVar = [
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
        """
        Create a events object with a UTCTimeObject as event creation info.
        """
        creation_info = CreationInfo(creation_time=obspy.UTCDateTime(time))
        event = Event(creation_info=creation_info)
        return Catalog(events=[event])

    # fixtures
    @pytest.fixture(params=times)
    def cat(self, request):
        """Create a events using value from time in the creation info"""
        return self.create_catalog(request.param)

    @pytest.fixture
    def cat2(self, cat):
        """Serialize events, then load in in again and return"""
        json = cat_to_dict(cat)
        return json_to_cat(json)

    # tests
    def test_equal(self, cat, cat2):
        """Ensure the catalogs are equal before and after serialization"""
        assert cat == cat2
