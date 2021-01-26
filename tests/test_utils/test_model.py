"""
Tests for the ObsPlusModel.
"""
import pickle

import glom
import obspy.core.event as ev
import pytest

import obsplus
from obsplus.exceptions import InvalidModelOperation, InvalidModelAttribute
from obsplus.events.schema import Catalog, Event, Origin
from obsplus.utils.model import ObsPlusModel, OperationTracker, ResourceIdentifier
from obsplus.utils.misc import register_func


@pytest.fixture()
def catalog_graph_dict():
    """Return the graph dict for catalog."""
    return Catalog.get_graph_dict()


class TestGraphDict:
    """Tests for getting the graph dictionary which defines relationships."""


    def test_expected_graph_keys(self, catalog_graph_dict):
        """ Ensure each of the models have a place in the graph. """
        for name in dir(eschema):
            if isinstance(getattr(eschema, name, None), ObsPlusModel):
                assert name in catalog_graph_dict

    def test_attr_id_ref(self, catalog_graph_dict):
        """
        Ensure the attr_id_ref, which tells us what kind of object the
        resource_id points to, is populated correctly on event.
        """
        event_dict = catalog_graph_dict['Event']
        ref_id_dict = event_dict['attr_id_ref']
        assert len(ref_id_dict)
        assert ref_id_dict['preferred_origin_id'] == 'Origin'
        assert ref_id_dict['preferred_magnitude_id'] == 'Magnitude'


class TestOperationTrackerBasic:
    """Tests for model attribute proxies."""
    proxy_list = []

    @pytest.fixture(scope='class')
    @register_func(proxy_list)
    def reference_id_proxy(self):
        """Return a proxy from a reference id"""
        return Event.preferred_origin_id

    @pytest.fixture(scope='class')
    @register_func(proxy_list)
    def list_id_proxy(self):
        """Return a proxy from a reference id"""
        return Catalog.events.preferred_origin_id

    @pytest.fixture(scope='class')
    @register_func(proxy_list)
    def time_id_proxy(self):
        """Return a proxy from a reference id"""
        return Origin.time

    @pytest.fixture(scope='class')
    @register_func(proxy_list)
    def chase_reference_id_proxy(self):
        """
        This should "chase" the attributes on the reference id the object
        points to.
        """
        return Event.preferred_origin_id.time

    @pytest.fixture(scope='class', params=proxy_list)
    def model_proxy(self, request):
        """meta-fixture to aggregate proxies."""
        return request.getfixturevalue(request.param)

    def test_get_proxy(self, model_proxy):
        """Ensure a proxy is returned from class level get_attrs. """
        assert isinstance(model_proxy, OperationTracker)
        assert str(model_proxy)

    def test_get_attr(self, list_id_proxy):
        """Tests for list ID proxy."""
        assert "Catalog.events.preferred_origin_id" in str(list_id_proxy)

    def test_or(self, list_id_proxy, reference_id_proxy):
        """Ensure or operator works"""
        out = list_id_proxy | reference_id_proxy
        assert ' | ' in str(out)
        # a second or is not currently allowed
        with pytest.raises(InvalidModelOperation, match='Only one'):
            out | out

    def test_get_item(self, reference_id_proxy):
        """Ensure get item works."""
        out = reference_id_proxy[0]
        assert "[0]" in str(out)

    def test_get_item_raises_on_non_int(self, reference_id_proxy):
        """Get item should only work on ints."""
        with pytest.raises(TypeError, match='must be an int'):
            reference_id_proxy['not an int']


class TestValidateModelOperationTracker:
    """Tests for validing trackers."""

    def test_invalid_path_raises(self, catalog_graph_dict):
        """Ensure a non-exist path raises. """
        with pytest.raises(InvalidModelAttribute, match='bob'):
            Catalog.events.bob.validate(catalog_graph_dict)

    def test_validate_or(self, catalog_graph_dict):
        """Ensure | which don't have the same shape fail."""
        first = Catalog.resource_id
        second = Event.picks[0]
        out = first | second
        with pytest.raises(InvalidModelOperation):
            out.validate(catalog_graph_dict)

    def test_or_then_attribute(self, catalog_graph_dict):
        """Ensure we can get the attributes from the result of |"""
        out = (Event.origins[0] | Event.preferred_origin_id).time
        out.validate(catalog_graph_dict)


class TestGlomSpec:
    """Ensure the glom spec can be fetched at all levels."""

    def test_gloom_spec(self):
        """Ensure the glom spec is correct"""

    def test_get_item(self):
        """Ensure a class"""








