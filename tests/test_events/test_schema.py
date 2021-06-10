"""
Tests for event schema.
"""
import copy
from typing import Sequence

import obspy
import obspy.core.event as ev
import pytest

import obsplus.events.schema as esc
import obsplus.structures.model
from obsplus.constants import NSLC
from obsplus.events.json import cat_to_dict
from pydantic import ValidationError


class TestResourceID:
    """Tests for the resource ID"""

    def test_null(self):
        """Ensure Null generates a resource ID a la ObsPy"""
        rid = obsplus.structures.model.ResourceIdentifier()
        assert isinstance(rid.id, str)
        assert len(rid.id)

    def test_defined_resource_id(self):
        """Ensure the defined resource_id sticks."""
        rid = str(ev.ResourceIdentifier())
        out = obsplus.structures.model.ResourceIdentifier(id=rid)
        assert out.id == rid


class TestWaveformID:
    """Test the waveform id object"""

    def test_seed_id(self):
        """Ensure the seed id is valid."""
        seed_id = "UU.TMU.01.HHZ"
        out = esc.WaveformStreamID(seed_string=seed_id)
        for name, value in zip(NSLC, seed_id.split(".")):
            assert getattr(out, f"{name}_code") == value


class TestConversions:
    """Tests for converting ObsPy catalogs to json."""

    def assert_lens_equal(self, obj1, obj2):
        """
        Transverse common attributes of obj1 and obj2, assert list-likes
        are equal.
        """
        # check sequences
        if isinstance(obj1, Sequence) and not isinstance(obj1, str):
            assert isinstance(obj2, Sequence)
            assert len(obj1) == len(obj2)
            # recurse
            for sub1, sub2 in zip(obj1, obj2):
                self.assert_lens_equal(sub1, sub2)
        # if working with non sequence
        else:
            # get overlapping attributes
            overlaps = set(dir(obj1)) & set(dir(obj2))
            for overlap in {x for x in overlaps if not x.startswith("_")}:
                # any attributes which are common and collections check
                sub1, sub2 = getattr(obj1, overlap), getattr(obj2, overlap)
                if isinstance(sub1, Sequence):
                    self.assert_lens_equal(sub1, sub2)

    def test_from_simple_obspy(self):
        """Test converting from a simple obspy catalog"""
        event = obspy.read_events()[0]
        pydantic_event = esc.Event.parse_obj(event)
        assert isinstance(pydantic_event, esc.Event)

    def test_from_obspy_catalog(self, test_catalog):
        """Ensure pydantic models can be generated from Obspy objects"""
        out = esc.Catalog.from_orm(test_catalog)
        assert isinstance(out, esc.Catalog)
        assert len(out.events) == len(test_catalog.events)
        self.assert_lens_equal(out, test_catalog)

    def test_from_json(self, test_catalog):
        """Ensure the catalog can be created from json."""
        catalog_dict = cat_to_dict(test_catalog)
        out = esc.Catalog.parse_obj(catalog_dict)
        assert isinstance(out, esc.Catalog)
        assert len(out.events) == len(catalog_dict["events"])

    def test_round_trip(self, test_catalog):
        """Test converting from pydantic models to ObsPy."""
        pycat = esc.Catalog.from_orm(test_catalog)
        out = pycat.to_obspy()
        assert out == test_catalog


class TestOrigin:
    """Tests for various validations on Origin object."""

    def test_origin_longitude(self):
        """Ensure origin longitude is coerced into correct range."""
        ori = ev.Origin(longitude=190.0)
        # this passes if no error is raised, should coerce long into correct range
        out = esc.Origin.from_orm(ori)
        assert out.longitude == -170

    def test_origin_latitude(self):
        """Bad latitude should raise an error, coercing them is ambigous."""
        ori = ev.Origin(latitude=99)
        with pytest.raises(ValidationError):
            esc.Origin.from_orm(ori)


class TestEvent:
    """Test for event model."""

    def test_resource_id(self):
        """Ensure the ResourceID gets created."""
        out = esc.Event()
        assert out.resource_id is not None

    def test_non_mutable_defaults(self):
        """
        Ensure appending to a default list in one instances doesn't effect others.
        """
        ev1 = esc.Event()
        ev2 = esc.Event()
        ev1.comments.append("bob")
        assert "bob" in ev1.comments
        assert not len(ev2.comments)

    def test_get_preferred_origin(self, model_catalog):
        """
        Test getting preferred origin when it exists
        """
        event = model_catalog.events[0]
        pref_origin = event.get_preferred_origin()
        assert pref_origin is not None
        assert isinstance(pref_origin, esc.Origin)
        assert str(pref_origin.resource_id) == str(event.preferred_origin_id)

    def test_get_preferred_magnitude(self, model_catalog):
        """
        Test getting preferred magnitude when it exists.
        """
        event = model_catalog.events[0]
        pref_origin = event.get_preferred_magnitude()
        assert pref_origin is not None
        assert isinstance(pref_origin, esc.Magnitude)
        assert str(pref_origin.resource_id) == str(event.preferred_magnitude_id)

    def test_get_preferred_focal_mechanism(self, model_catalog):
        """
        Test getting preferred focal mechanism. It doesn't exist.
        """
        event = model_catalog.events[0]
        focal_mech = event.get_preferred_focal_mechanism()
        assert focal_mech is None
        # but we can init a default focal mech
        focal_mech2 = event.get_preferred_focal_mechanism(new=True)
        assert isinstance(focal_mech2, esc.FocalMechanism)

    def test_preferred_wrong_id(self, model_catalog):
        """The preferred id is set but the object is missing raise ValueError."""
        event = copy.copy(model_catalog.events[0])
        event.preferred_magnitude_id = "Not a real ID, obviously"
        with pytest.raises(ValueError, match="No magnitude found"):
            event.get_preferred_magnitude()


class TestCatalog:
    """Tests for catalog."""

    def test_sequence(self, model_catalog):
        """Ensure the catalog behaves like a sequence."""
        cat = model_catalog
        # test len
        assert len(cat) == len(cat.events)
        # test iteration
        for e1, e2 in zip(cat, cat.events):
            assert e1 is e2
        # test subscription
        for num in range(len(cat.events)):
            e1, e2 = cat[num], cat.events[num]
            assert e1 is e2
