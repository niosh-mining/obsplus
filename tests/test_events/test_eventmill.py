"""Tests for EventMill."""
import numpy as np

import pandas as pd
import pytest

import obsplus
from obsplus import EventMill
from obsplus.exceptions import InvalidModelAttribute
from obsplus.utils.misc import register_func

event_dataframes = []


@pytest.fixture(scope="class")
@register_func(event_dataframes)
def event_dataframe(event_mill):
    """Return the event dataframe from event_mill."""
    return event_mill.get_df("events")


@pytest.fixture(scope='class')
@register_func(event_dataframes)
def pick_dataframe(event_mill):
    """Get the pick dataframe."""
    return event_mill.get_df('picks')


@pytest.fixture(scope='class', params=event_dataframes)
def eventmill_dataframe(request):
    """Meta fixture to parameterize all dataframes produced by eventmill."""
    return request.getfixturevalue(request.param)


class TestEventMillBasics:
    """Tests for the EventMill."""

    def test_init_with_catalog(self, bingham_catalog):
        """Ensure the mill can be initiated with a catalog."""
        mill = EventMill(bingham_catalog)
        assert isinstance(mill, EventMill)

    def test_init_with_json(self, bingham_catalog):
        """Ensure json structures also work (and get validated)."""
        json = obsplus.cat_to_json(bingham_catalog)
        mill = EventMill(json)
        assert isinstance(mill, EventMill)

    def test_str(self, event_mill):
        """Ensure a sensible str rep is available."""
        str_rep = str(event_mill)
        assert "Mill with spec of" in str_rep

    def test_lookup_homogeneous(self, event_mill):
        """Look up a resource id for objects of same type"""
        pick_df = event_mill.get_df("Pick")
        some_rids = pick_df.index.values[::20]
        # lookup multiple dataframes of the same type
        out, cls = event_mill.lookup(some_rids)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(some_rids)
        assert set(out.index.get_level_values("resource_id")) == set(some_rids)
        assert cls == "Pick"
        # lookup a single resource_id
        out, cls = event_mill.lookup(some_rids[0])
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 1
        assert set(out.index.get_level_values("resource_id")) == set(some_rids[:1])
        assert cls == "Pick"

    def test_lookup_missing(self, event_mill):
        """Test looking up a missing ID."""
        with pytest.raises(KeyError):
            event_mill.lookup("not a real_id")


class TestFillPreferred:
    """Tests for ensuring preferred values are set."""

    @pytest.fixture(scope="class")
    def missing_origin_id_mill(self, bingham_events):
        """Get a mill which has some preferred_origin_ids not set."""
        cat = bingham_events.copy()
        for num, eve in enumerate(cat):
            if num % 2 == 0:
                eve.preferred_origin_id = None
        return EventMill(cat)

    def test_fill_id(self, missing_origin_id_mill):
        """Ensure all preferred origin/mag ids are set."""
        out = missing_origin_id_mill.fill_preferred()
        event_df = out._table_dict["Event"]
        assert not event_df["preferred_origin_id"].isnull().any()


class TestGetChildren:
    """Tests for getting children of specific classes and attributes."""

    def test_get_picks(self, event_mill, bingham_events):
        """Get picks from event_mill."""
        pdf, _ = event_mill.get_children("Event", "picks")
        rids_mill = set(pdf.index)
        rids_cat = {
            str(pick.resource_id) for event in bingham_events for pick in event.picks
        }
        assert rids_mill == rids_cat

    def test_get_picks_with_df(self, event_mill, bingham_events):
        """Ensure a dataframe can be used to limit picks returned"""
        df = event_mill.get_df("Event").iloc[0:2]
        out, _ = event_mill.get_children("Event", "picks", df=df)
        # get expected pick ids
        expected = set()
        for event in bingham_events[:2]:
            for pick in event.picks:
                expected.add(str(pick.resource_id))
        assert set(out.index) == expected

    def test_bad_cls(self, event_mill):
        """Tests for class which don't exist."""
        with pytest.raises(KeyError, match="Unknown dataframe"):
            event_mill.get_children("NotAClass", "bad_attr")

    def test_bad_attr(self, event_mill):
        """Tests for accessing non-existent attributes"""
        with pytest.raises(InvalidModelAttribute, match="no model attributes"):
            event_mill.get_children("Event", "not_an_attr")


class TestGetParentIds:
    """Tests for finding parents of ids."""

    def test_no_level_no_limit(self, event_mill, bingham_events):
        """Tests for getting ids from default values."""
        catalog_id = str(bingham_events.resource_id)
        pick_ids = [str(pick.resource_id) for pick in bingham_events[0].picks]
        parent_ids = event_mill.get_parent_ids(pick_ids)
        # the result should simply be the catalog id
        assert (parent_ids == catalog_id).all()

    def test_up_one_level(self, event_mill, bingham_events):
        """Tests getting ids for one level of parents up."""
        event_id = str(bingham_events[0].resource_id)
        pick_ids = [str(x.resource_id) for x in bingham_events[0].picks]
        out = event_mill.get_parent_ids(pick_ids, level=1)
        assert (out == event_id).all()

    def test_target(self, event_mill, bingham_events):
        """Tests for stopping transverse at certain targets."""
        targets = {str(x.resource_id) for x in bingham_events}
        pick_ids = {
            str(pick.resource_id) for event in bingham_events for pick in event.picks
        }
        out = event_mill.get_parent_ids(pick_ids, targets=targets)
        assert set(out.values).issubset(targets)


class TestGetDF:
    """Tests for getting various forms of dataframes from EventMill."""


    def test_all_df(self, eventmill_dataframe):
        """Test all the dfs."""
        assert isinstance(eventmill_dataframe, pd.DataFrame)
        assert len(eventmill_dataframe)

    def test_get_contained_model(self, event_mill):
        """The name of a model should return the contained df."""
        out = event_mill.get_df("Event")
        assert isinstance(out, pd.DataFrame)
        assert len(out)

    def test_raise_on_unknown(self, event_mill):
        """Ensure unknown frames raise exception."""
        with pytest.raises(KeyError):
            event_mill.get_df(name="not_a_valid_dataframer_name")

    def test_get_event_dataframe(self, event_dataframe):
        """Tests for getting dataframes from mill."""
        assert isinstance(event_dataframe, pd.DataFrame)


class TestEventDataframe:
    """Tests specifically for the event dataframe."""

    @pytest.fixture(scope="class")
    def row_event_list(self, event_dataframe, bingham_events):
        """return a list of (event_df, event) for comparing row with event."""
        out = []
        for (_, row), event in zip(event_dataframe.iterrows(), bingham_events):
            out.append((row, event))
        return out

    def test_len(self, event_dataframe, bingham_catalog):
        """The dataframe should have one row for each event."""
        assert len(event_dataframe) == len(bingham_catalog)

    def test_ids_and_order(self, row_event_list):
        """The event order should remain unchanged."""
        for row, event in row_event_list:
            assert row["event_id"] == str(event.resource_id)

    def test_event_description(self, row_event_list):
        """Ensure the event descriptions match"""
        for row, event in row_event_list:
            if event.event_descriptions:
                assert row["event_description"] == event.event_descriptions[0].text

    def test_origin_info(self, row_event_list):
        """Ensure the origin info (time, location) match."""
        for row, event in row_event_list:
            ori = event.preferred_origin()
            assert np.isclose(row["event_longitude"], ori.longitude)
            assert np.isclose(row["event_latitude"], ori.latitude)
            assert np.isclose(row["event_depth"], ori.depth)
            etime1 = obsplus.utils.time.to_utc(ori.time)
            etime2 = obsplus.utils.time.to_utc(row["event_time"])
            assert np.isclose(float(etime1), float(etime2))
