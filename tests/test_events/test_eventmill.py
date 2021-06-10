"""Tests for EventMill."""
import numpy as np
import pandas as pd
import pytest

import obsplus
from obsplus import EventMill


@pytest.fixture(scope="class")
def event_dataframe(event_mill):
    """Return the event dataframe from event_mill."""
    return event_mill.get_df("events")


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


class TestGetReferredObject:
    """Tests for getting the referred object based on str ID."""

    def test_get_referred_object_address(self, event_mill, bingham_events):
        """Ensure the address of the referred object can be found."""
        # first get first event address
        eid = str(bingham_events[0].resource_id)
        address = event_mill.get_referred_address(eid)
        assert address == ("events", 0)
        # next try pick
        eid = str(bingham_events[0].picks[1].resource_id)
        address = event_mill.get_referred_address(eid)
        assert address == ("events", 0, "picks", 1)


class TestGetDF:
    """Tests for getting various forms of dataframes from EventMill."""

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
            etime = obsplus.utils.time.to_timedelta64(ori.time)
            assert np.isclose(row["event_time"], etime)
