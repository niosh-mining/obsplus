"""
Tests for the ObsPlusModel.
"""
from typing_extensions import Literal

import pytest
import obsplus.events.schema as eschema
from obsplus.exceptions import InvalidModelAttribute
from obsplus.events.schema import Catalog, Event, Origin
from obsplus.structures.model import ObsPlusModel, _SpecGenerator, spec_callable
from obsplus.utils.misc import register_func


@pytest.fixture()
def catalog_graph_dict():
    """Return the graph dict for catalog."""
    return Catalog.get_obsplus_schema()


class TestGraphDict:
    """Tests for getting the graph dictionary which defines relationships."""

    def test_expected_graph_keys(self, catalog_graph_dict):
        """Ensure each of the models have a place in the graph."""
        for name in dir(eschema):
            if isinstance(getattr(eschema, name, None), ObsPlusModel):
                assert name in catalog_graph_dict

    def test_attr_id_ref(self, catalog_graph_dict):
        """
        Ensure the attr_id_ref, which tells us what kind of object the
        resource_id points to, is populated correctly on event.
        """
        event_dict = catalog_graph_dict["Event"]
        ref_id_dict = event_dict["attr_id_ref"]
        assert len(ref_id_dict)
        assert ref_id_dict["preferred_origin_id"] == "Origin"
        assert ref_id_dict["preferred_magnitude_id"] == "Magnitude"


class TestSpecGenerator:
    """Tests for generating tree specs."""

    proxy_list = []

    @pytest.fixture(scope="class")
    @register_func(proxy_list)
    def reference_id_proxy(self):
        """Return a proxy from a reference id"""
        return Event.preferred_origin_id

    @pytest.fixture(scope="class")
    @register_func(proxy_list)
    def time_id_proxy(self):
        """Return a proxy from a reference id"""
        return Origin.time

    @pytest.fixture(scope="class")
    @register_func(proxy_list)
    def chase_reference_id_proxy(self):
        """
        This should "chase" the attributes on the reference id the object
        points to.
        """
        return Event.preferred_origin_id.time

    @pytest.fixture(scope="class", params=proxy_list)
    def model_proxy(self, request):
        """meta-fixture to aggregate proxies."""
        return request.getfixturevalue(request.param)

    def test_operation_tracker_basic_attribute(self):
        """Ensure an operation tracker is returned from a model."""
        time = Origin.time
        assert isinstance(time, _SpecGenerator)

    def test_preferred(self):
        """Tests for supporting preferred operator."""
        pref_magnitude = Event._preferred_magnitude
        assert isinstance(pref_magnitude, _SpecGenerator)

    def test_parent_preserved(self):
        """Ensure parent model reference is preserved."""
        mag = Event.magnitudes
        assert hasattr(mag, "parent_model")
        assert mag.parent_model == Event
        amp = mag.amplitude
        assert amp.parent_model == Event
        pick = amp.pick_id._referred_object
        assert pick.parent_model == Event

    def test_function(self):
        """Tests for calling  functions."""
        mag = Origin.preferred_origin().mag
        breakpoint()

    @pytest.mark.xfail
    def test_track_schema(self):
        """
        The operation tracker should know where it is in the schema
        and raise attribute errors if a non-existent attr is requested.
        """
        event = Catalog.events[0]
        with pytest.raises(InvalidModelAttribute, match="not_an_attribute"):
            event.not_an_attribute
        with pytest.raises(InvalidModelAttribute, match="still_wrong"):
            event.picks.still_wrong

        # but the special operations should work when they are supported
