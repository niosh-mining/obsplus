"""
Tests for the ObsPlusModel.
"""
import pytest

from obsplus.events.schema import Catalog, Event, Origin
from obsplus.exceptions import InvalidModelAttribute
from obsplus.structures.model import _SpecGenerator
from obsplus.utils.misc import register_func


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
        pref_magnitude = Event.preferred_magnitude_id
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
