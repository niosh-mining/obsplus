"""
Tests for the big events interface.
"""

import obspy
import obspy.core.event as ev
import pytest

import obsplus
from obsplus.utils.misc import yield_obj_parent_attr


@pytest.fixture(scope="class")
def catalog():
    """ load the Crandal events """
    return obsplus.load_dataset("crandall_test").event_client


@pytest.fixture(scope="class")
def bigcat(catalog):
    """ init a big events from the crandal canyon dataset. """
    return obsplus.BigCatalog.from_catalog(catalog)


class TestCatalogInterface:
    """ The big events should look and feel like a normal events. """

    def prune_catalog(self, catalog):
        """recurse a events and set all attrs that eval to False to None.
        This is needed to overcome some Catalog oddities to fairly compare two
        catalogs."""
        skips = (obspy.UTCDateTime, ev.ResourceIdentifier)
        cat = catalog.copy()
        for obj, parent, attr in yield_obj_parent_attr(cat):
            if isinstance(obj, skips):
                continue
            for item, val in obj.__dict__.items():
                setattr(obj, item, val or None)
        return cat

    def test_get_events(self, bigcat, catalog):
        """ Getting events from the events should be equal to normal events """
        for ev1, ev2 in zip(bigcat, catalog):
            event1, event2 = self.prune_catalog(ev1), self.prune_catalog(ev2)
            assert event1 == event2
            assert event1 is not event2


class TestDataFrames:
    """ tests for returning dataframes. """

    @pytest.fixture
    def arrivals(self, bigcat):
        """ return the arrivals from the big events """
        return bigcat.get_arrivals(join_externals=True)

    def test_arrivals(self, arrivals, catalog):
        """ ensure a dataframe is returned and contains all info. """
        # TODO start here


class TestInternalDataFrames:
    def test_parent_and_event_id(self, bigcat):
        """ ensure parent and event ids are in non-event tables """
        for name, df in bigcat._dfs.items():
            if name in {"Event", "ID"}:
                continue
            # ensure parent id is defined in all non-event tables
            assert (df["_parent_id_"].astype(bool)).all()
            # ensure event id is also defined
            assert (df["_event_id_"].astype(bool)).all()
