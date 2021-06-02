"""Tests for EventMill."""
import obsplus
from obsplus import EventMill

import pytest


@pytest.fixture()
def event_mill():
    """Create the event mill from a json dict."""


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


class TestGetDF:
    """Tests for getting various forms of dataframes from EventMill."""

    def test_get_event_dataframe(self):
        """Tests for getting dataframes from mill."""
        pass
