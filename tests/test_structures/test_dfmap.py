"""
Tests for dfmap.
"""

import pytest

from obsplus.structures.dfmap import DFMap


class TestSubclassDFMap:
    """Tests to ensure subclassing works."""

    def test_must_have_model(self):
        """A subclass must have a model."""

        with pytest.raises(AttributeError):

            class BadMapping(DFMap):
                """This mapping has no _model attr."""

        # with pytest.raises()
