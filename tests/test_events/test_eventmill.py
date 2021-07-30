"""Tests for EventMill."""
import numpy as np
import obspy
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

import obsplus
from obsplus import EventMill
from obsplus.exceptions import InvalidModelAttribute
from obsplus.utils.misc import register_func

event_dataframes = []
eventmills = ["bing_eventmill"]


@pytest.fixture(scope="class")
@register_func(event_dataframes)
def event_dataframe(bing_eventmill):
    """Return the event dataframe from event_mill."""
    return bing_eventmill.get_df("events")


@pytest.fixture(scope="class")
@register_func(event_dataframes)
def pick_dataframe(bing_eventmill):
    """Get the pick dataframe."""
    return bing_eventmill.get_df("picks")


@pytest.fixture(scope="class", params=event_dataframes)
def eventmill_dataframe(request):
    """Meta fixture to parameterize all dataframes produced by eventmill."""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="class")
def bingham_event_missing_preferred_origin(bingham_events):
    """return the bingham event missing preferred origin."""
    cat = bingham_events.copy()
    for num, eve in enumerate(cat):
        if num % 2 == 0:
            eve.preferred_origin_id = None
    return cat


@pytest.fixture(scope="class")
def missing_origin_id_mill(bingham_event_missing_preferred_origin):
    """Get a mill which has some preferred_origin_ids not set."""
    # shuffle events just to make sure there isn't an order dependence
    cat = bingham_event_missing_preferred_origin
    cat.events = sorted(cat.events, key=lambda x: str(x.resource_id))
    out = EventMill(bingham_event_missing_preferred_origin)
    _ = out.get_summary_df()
    return out


@pytest.fixture(scope="class")
@register_func(eventmills)
def eventmill_empty():
    """Return an eventmill with no events."""
    cat = obspy.Catalog()
    return EventMill(cat)


@pytest.fixture(scope="class")
@register_func(eventmills)
def eventmill_one_event(bingham_events):
    """Return an eventmill with one event."""
    cat = bingham_events.copy()
    cat.events = [cat.events[0]]
    return EventMill(cat)


@pytest.fixture(scope="class", params=eventmills)
def event_mill(request):
    """Meta fixture to collect all eventmills."""
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

    def test_str(self, bing_eventmill):
        """Ensure a sensible str rep is available."""
        str_rep = str(bing_eventmill)
        assert "Mill with spec of" in str_rep

    def test_copy(self, bing_eventmill):
        """Ensure copying eventmill creates a new mill."""
        out = bing_eventmill.copy()
        # ensure managed dataframes are not identical
        for table_name in out._get_table_dict():
            df1 = out._get_table_dict()[table_name]
            df2 = bing_eventmill._get_table_dict()[table_name]
            assert df1.equals(df2)
            assert df1 is not df2

    def test_scoping(self, bing_eventmill):
        """Ensure scoping is listed."""
        df = bing_eventmill.get_df("__structure__")
        no_scope = df["scope_id"].isnull()
        has_scope = ~no_scope
        assert (has_scope.sum() / no_scope.sum()) > 0.9

    def test_summary(self, bing_eventmill, bingham_events):
        """Ensure the summary dataframe is populated."""
        df = bing_eventmill.get_df("__summary__")
        assert len(df) == len(bingham_events)
        # ensure order is preserved
        for (rid, ser), event in zip(df.iterrows(), bingham_events):
            assert rid == str(event.resource_id)


class TestLookUp:
    """Tests for looking up objects by ID."""

    def test_lookup_homogeneous(self, bing_eventmill):
        """Look up a resource id for objects of same type"""
        pick_df = bing_eventmill.get_df("Pick")
        some_rids = pick_df.index.values[::20]
        # lookup multiple dataframes of the same type
        out, cls = bing_eventmill.lookup(some_rids)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(some_rids)
        assert set(out.index.get_level_values("resource_id")) == set(some_rids)
        assert cls == "Pick"
        # lookup a single resource_id
        out, cls = bing_eventmill.lookup(some_rids[0])
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 1
        assert set(out.index.get_level_values("resource_id")) == set(some_rids[:1])
        assert cls == "Pick"

    def test_lookup_missing(self, bing_eventmill):
        """Test looking up a missing ID."""
        with pytest.raises(KeyError):
            bing_eventmill.lookup("not a real_id")

    def test_empty_ids(self, bing_eventmill):
        """Ensure empty ids returns emtpy df and NA for class name"""
        df, cls = bing_eventmill.lookup([])
        assert not len(df)
        assert pd.isnull(cls)


class TestGetChildren:
    """Tests for getting children of specific classes and attributes."""

    def test_get_picks(self, bing_eventmill, bingham_events):
        """Get picks from event_mill."""
        pdf, _ = bing_eventmill.get_children("Event", "picks")
        rids_mill = set(pdf.index)
        rids_cat = {
            str(pick.resource_id) for event in bingham_events for pick in event.picks
        }
        assert rids_mill == rids_cat

    def test_get_picks_with_df(self, bing_eventmill, bingham_events):
        """Ensure a dataframe can be used to limit picks returned"""
        df = bing_eventmill.get_df("Event").iloc[0:2]
        out, _ = bing_eventmill.get_children("Event", "picks", df=df)
        # get expected pick ids
        expected = set()
        for event in bingham_events[:2]:
            for pick in event.picks:
                expected.add(str(pick.resource_id))
        assert set(out.index) == expected

    def test_bad_cls(self, bing_eventmill):
        """Tests for class which don't exist."""
        with pytest.raises(KeyError, match="Unknown dataframe"):
            bing_eventmill.get_children("NotAClass", "bad_attr")

    def test_bad_attr(self, bing_eventmill):
        """Tests for accessing non-existent attributes"""
        with pytest.raises(InvalidModelAttribute, match="no model attributes"):
            bing_eventmill.get_children("Event", "not_an_attr")

    def test_bad_ids(self):
        """Tests for getting child"""


class TestFillPreferred:
    """Tests for ensuring preferred values are set."""

    def test_fill_id(
        self,
        missing_origin_id_mill,
        bingham_event_missing_preferred_origin,
    ):
        """Ensure all preferred origin/mag ids are set."""
        cat_dict = {
            str(e.resource_id): e for e in bingham_event_missing_preferred_origin
        }
        # ensure no ids are empty
        out = missing_origin_id_mill.fill_preferred()
        event_df = out.get_df("Event")
        assert not event_df["preferred_origin_id"].isnull().any()
        # make sure the same event_ids map to the original preferred
        ser = event_df["preferred_origin_id"]
        for event_id, origin_id in ser.items():
            event = cat_dict[event_id]
            if event.preferred_origin_id is not None:
                assert str(event.preferred_origin_id) == origin_id
            else:
                last_origin = event.origins[-1]
                assert str(last_origin.resource_id) == origin_id


class TestGetParentIds:
    """Tests for finding parents of ids."""

    def test_no_level_no_limit(self, bing_eventmill, bingham_events):
        """Tests for getting ids from default values."""
        catalog_id = str(bingham_events.resource_id)
        pick_ids = [str(pick.resource_id) for pick in bingham_events[0].picks]
        parent_ids = bing_eventmill.get_parent_ids(pick_ids)
        # the result should simply be the catalog id
        assert (parent_ids == catalog_id).all()

    def test_up_one_level(self, bing_eventmill, bingham_events):
        """Tests getting ids for one level of parents up."""
        event_id = str(bingham_events[0].resource_id)
        pick_ids = [str(x.resource_id) for x in bingham_events[0].picks]
        out = bing_eventmill.get_parent_ids(pick_ids, level=1)
        assert (out == event_id).all()

    def test_target(self, bing_eventmill, bingham_events):
        """Tests for stopping transverse at certain targets."""
        targets = {str(x.resource_id) for x in bingham_events}
        pick_ids = {
            str(pick.resource_id) for event in bingham_events for pick in event.picks
        }
        out = bing_eventmill.get_parent_ids(pick_ids, targets=targets)
        assert set(out.values).issubset(targets)


class TestGetDF:
    """Tests for getting various forms of dataframes from EventMill."""

    def test_all_df(self, eventmill_dataframe):
        """Test all the dfs."""
        assert isinstance(eventmill_dataframe, pd.DataFrame)
        assert len(eventmill_dataframe)

    def test_get_contained_model(self, bing_eventmill):
        """The name of a model should return the contained df."""
        out = bing_eventmill.get_df("Event")
        assert isinstance(out, pd.DataFrame)
        assert len(out)

    def test_raise_on_unknown(self, bing_eventmill):
        """Ensure unknown frames raise exception."""
        with pytest.raises(KeyError):
            bing_eventmill.get_df(name="not_a_valid_dataframer_name")

    def test_get_event_dataframe(self, event_dataframe):
        """Tests for getting dataframes from mill."""
        assert isinstance(event_dataframe, pd.DataFrame)

    def test_empty_str_default_resource_id(self, bing_eventmill):
        """Ensure missing ids are empty strings rather than 'None' or 'nan'"""
        df = bing_eventmill.get_df("StationMagnitude")
        for col in ["amplitude_id", "method_id", "origin_id"]:
            ser = df[col]
            is_empty = ser.isnull()
            len_gt_40 = ser.str.len() > 40
            assert (is_empty | len_gt_40).all()

    def test_filter_on_scope_existing_table(self, bing_eventmill, bingham_events):
        """Ensure any supported scope kwargs can filter dfs."""
        sub = bingham_events.get_events(minmagnitude=1)
        pick_ids = {str(p.resource_id) for e in sub for p in e.picks}
        df = bing_eventmill.get_df("Pick", minmagnitude=1)
        assert pick_ids == set(df.index)

    def test_filter_on_dataframe_extractor_df(self, bing_eventmill, bingham_events):
        """Tests for filtering w/ dataframers"""
        sub = bingham_events.get_events(minmagnitude=1)
        pick_ids = {str(p.resource_id) for e in sub for p in e.picks}
        out = bing_eventmill.get_df("picks", minmagnitude=1)
        assert set(out.index) == pick_ids


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
            assert np.isclose(row["longitude"], ori.longitude)
            assert np.isclose(row["latitude"], ori.latitude)
            assert np.isclose(row["depth"], ori.depth)
            etime1 = obsplus.utils.time.to_utc(ori.time)
            etime2 = obsplus.utils.time.to_utc(row["time"])
            assert np.isclose(float(etime1), float(etime2))

    def test_one_event(self, bingham_events):
        """Test getting event_df with a single event."""
        events = bingham_events.copy()
        events.events = list(events[:1])
        mill = EventMill(events)
        df = mill.get_summary_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


class TestToModel:
    """Tests converting mills back to models."""

    @pytest.fixture()
    def model_from_mill(self, bing_eventmill):
        """Convert the event_mill to model."""
        mod = bing_eventmill.to_model()
        return mod

    def test_model_conversion(self, model_from_mill, bingham_model, bing_eventmill):
        """ensure the model was losslessly converted."""
        cat1 = model_from_mill.to_obspy()
        cat2 = bingham_model.to_obspy()
        # event resource ids should be unique
        assert len(cat1) == len({str(x.resource_id) for x in cat1})
        assert len(cat1) == len(cat2)
        assert cat1 == cat2


class TestGetEvents:
    """
    Tests for extracting events from EventMill.

    Note: since EventMill.get_events just uses well tested functions from
    events.get_events we don't really need many tests here.
    """

    def test_simple(self, bing_eventmill):
        """Test get events with no params"""
        out = bing_eventmill.get_events()
        assert isinstance(out, obspy.Catalog)
        assert len(out) == len(bing_eventmill.get_df("Event"))

    def test_query_by_event_description(self, bing_eventmill, bingham_events):
        """Query by event description."""
        expected = defaultdict(list)
        for event in bingham_events:
            if not event.event_descriptions:
                continue
            expected[event.event_descriptions[0].text].append(event)

        out1 = bing_eventmill.get_events(event_description="LR")
        assert len(out1) == len(expected["LR"])

        out2 = bing_eventmill.get_events(event_description={"LR", "RQ"})
        assert len(out2) == (len(expected["LR"]) + len(expected["RQ"]))


class TestToParquet:
    """Tests for dumping a mill to a parquet directory"""

    @pytest.fixture(scope="class")
    def saved_mill(self, event_mill, tmp_path_factory):
        """Save the mill and return path."""
        path = tmp_path_factory.mktemp("saved_mill") / "mill.zip"
        event_mill.to_parquet(path)
        return path

    @pytest.fixture(scope="class")
    def loaded_mill(self, saved_mill):
        """Load mill into memory."""
        return EventMill.from_parquet(saved_mill)

    def test_file_created(self, saved_mill):
        """Ensure the expected file now exists."""
        assert Path(saved_mill).exists()

    def test_loaded_mill(self, loaded_mill, event_mill):
        """Ensure the mill is loaded and equal to input mill."""
        assert str(loaded_mill) == str(event_mill)
        # TODO need to implement proper equality checks for Mills
