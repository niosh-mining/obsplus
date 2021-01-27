"""
A module for testing the mill.
"""
import pytest

import obsplus
import obsplus.events.schema as schema
from obsplus.structures.mill import Mill, DFCatalogMapping


@pytest.fixture(scope="class")
def event_mill():
    """Init a mill from an event."""
    cat = obsplus.load_dataset("bingham_test").event_client.get_events()
    mill = Mill(cat, schema.CatalogSchema)
    return mill


class TestDFMapper:
    """tests for mapping dataframes to tree structures."""


class TestMillBasics:
    """Tests for the basics of the mill."""

    def test_str(self, event_mill):
        """Ensure a sensible str rep is available."""
        breakpoint()
        str_rep = str(event_mill)
