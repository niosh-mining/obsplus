"""
A module for testing the mill.
"""
import pandas as pd
import pytest


class TestDFMapper:
    """tests for mapping dataframes to tree structures."""


class TestMillBasics:
    """Tests for the basics of the mill."""

    def test_str(self, event_mill):
        """Ensure a sensible str rep is available."""
        str_rep = str(event_mill)
        assert "Mill with spec of" in str_rep

    def test_lookup_homogeneous(self, event_mill):
        """Look up a resource id for objects of same type"""
        pick_df = event_mill._df_dicts["Pick"]
        some_rids = pick_df.index[::20]
        # lookup multiple dataframes of the same type
        out = event_mill.lookup(some_rids)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(some_rids)
        assert set(out.index) == set(some_rids)
        # lookup a single resource_id
        out = event_mill.lookup(some_rids[0])
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 1
        assert set(out.index) == set(some_rids[:1])

    def test_lookup_missing(self, event_mill):
        """Test looking up a missing ID."""
        with pytest.raises(KeyError):
            event_mill.lookup("not a real_id")
