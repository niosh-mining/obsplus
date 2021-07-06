"""
Tests for dfmap.
"""

import pytest

import obsplus.events.schema as eschema
from obsplus.exceptions import InvalidModelAttribute
from obsplus.structures.dataframer import DataFramer, model_operator


class TestSubclassDFMap:
    """Tests to ensure subclassing works."""

    def test_must_have_model(self):
        """A subclass must have a model."""

        with pytest.raises(AttributeError, match="missing attributes"):

            class BadMapping(DataFramer):
                """This mapping has no _model attr."""

    def test_parent_valid(self):
        """A subclass should be able to reference parents."""

        class FramerWithParent(DataFramer):
            """A framer which specifies an attribute which has a parent"""

            _model = eschema.Pick
            event_id = _model.parent().resource_id

    def test_wrong_base_raises(self):
        """A subclass which doesn't stem from model should raise"""

        with pytest.raises(InvalidModelAttribute):

            class AnotherBadFramer(DataFramer):
                """This one is bad because not all attrs are based on _model"""

                _model = eschema.Origin

                origin_time = _model.time  # this is ok
                event_id = _model.parent().resource_id  # also ok
                pick_time = eschema.Pick.time  # bad, not based on _model

    def test_model_operator_bad_parameters(self):
        """A subclass with an operator that has an unsupported parameter."""

        with pytest.raises(TypeError, match="not supported parameter names"):

            class FramerBadOperator(DataFramer):
                """A framer with a bad operator."""

                _model = eschema.Origin

                @model_operator
                def _bad_operator(self, unsupported_parameter):
                    """We can't know what unsupported_parameter is."""
