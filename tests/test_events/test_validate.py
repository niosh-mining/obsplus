from os.path import join

import obsplus
import obsplus.events.validate
import obspy
import pytest
from obsplus import validate_catalog
from obspy.core.event import ResourceIdentifier

CAT1_PATH = join(pytest.test_data_path, "qml_files", "2017-01-16T01-15-13-8a42f.xml")


# ----------------- module level fixtures


@pytest.fixture(scope="function")
def cat1():
    """ return a copy of events 1"""
    cat = obspy.read_events(CAT1_PATH)
    validate_catalog(cat)
    cat[0].focal_mechanisms.append(obspy.core.event.FocalMechanism())
    return cat


# ---------------- tests


class TestValidateCatalog:
    """ ensure check events properly puts each event into a good state for
    processing (IE all resource_id are attached, preferreds are set, there
    are no resource_ids referring to non-existent events, etc."""

    # helper functions
    def preferreds_are_set(self, cat):
        for eve in cat:
            if len(eve.origins):
                assert eve.preferred_origin_id is not None
                assert eve.preferred_origin() is not None
            if len(eve.magnitudes):
                assert eve.preferred_magnitude_id is not None
                assert eve.preferred_magnitude() is not None
            if len(eve.focal_mechanisms):
                assert eve.preferred_focal_mechanism_id is not None
                assert eve.preferred_focal_mechanism() is not None

    # fixtures

    @pytest.fixture()
    def cat1_multiple_resource_ids(self, cat1):
        """ copy the origins on the event so there are duplicate resource ids
        within one events """
        cat = cat1.copy()
        cat[0].origins = cat1[0].origins + cat[0].origins
        return cat

    @pytest.fixture()
    def cat1_cleared_preferreds(self, cat1):
        """ clear the preferred values of the events, return events """
        validate_catalog(cat1)
        cat = cat1.copy()
        cat[0].preferred_origin_id = None
        cat[0].preferred_magnitude_id = None
        cat[0].preferred_focal_mechanism_id = None
        return cat

    @pytest.fixture()
    def cat1_preferred_cache_empty(self, cat1):
        """ Set each of the preferreds to the first in the list.
        Then monkey patch the resource_id_weak_dict.
        """
        # copy events
        cat = cat1.copy()
        eve = cat[0]

        # monkey patch resource_id dict
        mangled_name = "_ResourceIdentifier__resource_id_weak_dict"

        if not hasattr(ResourceIdentifier, mangled_name):
            pytest.skip("obsolete tests")

        rid = getattr(ResourceIdentifier, mangled_name)
        rid_type = type(rid)
        new = rid_type()
        setattr(ResourceIdentifier, mangled_name, new)

        # set preferreds to first in the list
        for name in ["origin", "magnitude", "focal_mechanism"]:
            preferred_id_name = "preferred_" + name + "_id"
            first_obj = getattr(eve, name + "s")[0]
            first_rid = first_obj.resource_id
            new_id = ResourceIdentifier(first_rid.id)
            setattr(eve, preferred_id_name, new_id)

        # ensure the events state is correct
        assert cat[0].preferred_origin_id is not None
        assert cat[0].preferred_magnitude_id is not None
        assert cat[0].preferred_focal_mechanism_id is not None
        assert cat[0].preferred_origin() is None
        assert cat[0].preferred_magnitude() is None
        assert cat[0].preferred_focal_mechanism() is None
        yield cat
        # restore old resource dict
        setattr(ResourceIdentifier, mangled_name, new)

    @pytest.fixture()
    def cat1_bad_arrival_pick_id(self, cat1):
        cat = cat1.copy()
        rid = ResourceIdentifier()
        cat[0].origins[0].arrivals[0].pick_id = rid
        return cat

    @pytest.fixture()
    def cat1_none_arrival_pick_id(self, cat1):
        cat = cat1.copy()
        cat[0].origins[0].arrivals[0].pick_id = None
        return cat

    @pytest.fixture()
    def cat1_no_pick_phase_hints(self, cat1):
        """ clear the phase hints in the first pick """
        cat = cat1.copy()
        cat[0].picks[0].phase_hint = None
        return cat

    @pytest.fixture()
    def cat1_no_pick_waveform_id(self, cat1):
        """ clear the phase hints in the first pick """
        cat = cat1.copy()
        cat[0].picks[0].waveform_id = None
        return cat

    # tests
    def test_pcat1_cleared_preferreds(self, cat1_cleared_preferreds):
        """ cleared preferreds should be reset to last in list"""
        cat = cat1_cleared_preferreds
        validate_catalog(cat)
        self.preferreds_are_set(cat)
        # make sure it is the last ones in the list
        ev = cat[0]
        if len(ev.origins):
            assert ev.preferred_origin() == ev.origins[-1]
        if len(ev.magnitudes):
            assert ev.preferred_magnitude() == ev.magnitudes[-1]
        if len(cat[0].focal_mechanisms):
            assert ev.preferred_focal_mechanism() == ev.focal_mechanisms[-1]

    def test_cat1_preferred_cache_empty(self, cat1_preferred_cache_empty):
        """ ensure preferred still point to correct (not last) origins/mags """
        cat = cat1_preferred_cache_empty
        validate_catalog(cat)
        self.preferreds_are_set(cat)
        # ensure the preferred are still the first
        if len(cat[0].origins):
            first_origin = cat[0].origins[0]
            assert cat[0].preferred_origin() == first_origin
        if len(cat[0].magnitudes):
            first_magnitude = cat[0].magnitudes[0]
            assert cat[0].preferred_magnitude() == first_magnitude
        if len(cat[0].focal_mechanisms):
            first_mech = cat[0].focal_mechanisms[0]
            assert cat[0].preferred_focal_mechanism() == first_mech

    def test_bad_arrival_pick_id_raises(self, cat1_bad_arrival_pick_id):
        """ make sure a bad pick_id in arrivals raises assertion error """
        with pytest.raises(AssertionError):
            validate_catalog(cat1_bad_arrival_pick_id)

    def test_duplicate_objects_raise(self, cat1_multiple_resource_ids):
        """ make sure an assertion error is raised on cat2 as it's resource
        ids are not unique """
        with pytest.raises(AssertionError):
            validate_catalog(cat1_multiple_resource_ids)

    def test_empty_phase_hint_raises(self, cat1_no_pick_phase_hints):
        """ ensure raises if any phase hints are undefined """
        with pytest.raises(AssertionError):
            validate_catalog(cat1_no_pick_phase_hints)

    def test_empty_pick_wid_raises(self, cat1_no_pick_waveform_id):
        """ ensure raise if any waveform ids are empty on picks """
        with pytest.raises(AssertionError):
            validate_catalog(cat1_no_pick_waveform_id)

    def test_none_in_arrival_pick_id_fails(self, cat1_none_arrival_pick_id):
        """ make sure if an arrival has a None pick validate raises """
        with pytest.raises(AssertionError):
            validate_catalog(cat1_none_arrival_pick_id)

    def test_works_with_event(self, cat1):
        """ ensure the method can also be called on an event """
        validate_catalog(cat1[0])


class TestAddValidator:
    """ ensure validators can be added """

    counter1 = 0

    @pytest.fixture()
    def add_validator(self):
        """ temp add a validator that increments the counter """

        @obsplus.catalog_validator
        def tick_counter1(catalog):
            self.counter1 += 1

        yield
        obsplus.events.validate.CATALOG_VALIDATORS.remove(tick_counter1)

    @pytest.fixture
    def val_cat(self, cat1, add_validator):
        """ validate the events exactly once, return result """
        return validate_catalog(cat1)

    # tests
    def test_registered_function_ran_once(self, val_cat):
        """ ensure the ticker went up once """
        assert self.counter1 == 1
        assert isinstance(val_cat, obspy.Catalog)
