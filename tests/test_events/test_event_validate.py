"""
Tests for validation logic.
"""
import copy

import obspy
import obspy.core.event as ev
import pytest
from obspy.core.event import ResourceIdentifier, WaveformStreamID, TimeWindow

import obsplus
import obsplus.events.validate
from obsplus import validate_catalog
from obsplus.utils.misc import yield_obj_parent_attr


# ----------------- module level fixtures


@pytest.fixture(scope="function")
def cat1(event_cache):
    """return a copy of events 1"""
    cat = event_cache["2017-01-16T01-15-13-8a42f.xml"].copy()
    validate_catalog(cat)
    cat[0].focal_mechanisms.append(obspy.core.event.FocalMechanism())
    return cat


# ---------------- tests


class TestValidateCatalog:
    """
    Ensure check events properly puts each event into a good state for
    processing (IE all resource_id are attached, preferreds are set, there
    are no resource_ids referring to non-existent events, etc.
    """

    # helper functions
    def preferred_ids_are_set(self, cat):
        """Return True if preferred ids are set."""
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
        """
        Copy the origins on the event so there are duplicate resource ids
        within one events.
        """
        cat = cat1.copy()
        cat[0].origins = cat1[0].origins + cat[0].origins
        return cat

    @pytest.fixture()
    def cat1_cleared_preferreds(self, cat1):
        """clear the preferred values of the events, return events"""
        validate_catalog(cat1)
        cat = cat1.copy()
        cat[0].preferred_origin_id = None
        cat[0].preferred_magnitude_id = None
        cat[0].preferred_focal_mechanism_id = None
        return cat

    @pytest.fixture()
    def cat1_preferred_cache_empty(self, cat1):
        """
        Set each of the preferreds to the first in the list. Then monkey
        patch the resource_id_weak_dict.
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
        """Create a catalog with a bad arrival (no id)"""
        cat = cat1.copy()
        rid = ResourceIdentifier()
        cat[0].origins[0].arrivals[0].pick_id = rid
        return cat

    @pytest.fixture()
    def cat1_none_arrival_pick_id(self, cat1):
        """Return a catalog with arrival with no pick_id."""
        cat = cat1.copy()
        cat[0].origins[0].arrivals[0].pick_id = None
        return cat

    @pytest.fixture()
    def cat1_no_pick_phase_hints(self, cat1):
        """clear the phase hints in the first pick"""
        cat = cat1.copy()
        cat[0].picks[0].phase_hint = None
        return cat

    @pytest.fixture()
    def cat1_no_pick_waveform_id(self, cat1):
        """clear the phase hints in the first pick"""
        cat = cat1.copy()
        cat[0].picks[0].waveform_id = None
        return cat

    @pytest.fixture
    def cat_nullish_nslc_codes(self, cat1):
        """Create several picks with nullish location codes."""
        cat1[0].picks[0].waveform_id.location_code = "--"
        cat1[0].picks[1].waveform_id.location_code = None
        return validate_catalog(cat1)

    # tests
    def test_pcat1_cleared_preferreds(self, cat1_cleared_preferreds):
        """cleared preferreds should be reset to last in list"""
        cat = cat1_cleared_preferreds
        validate_catalog(cat)
        self.preferred_ids_are_set(cat)
        # make sure it is the last ones in the list
        ev = cat[0]
        if len(ev.origins):
            assert ev.preferred_origin() == ev.origins[-1]
        if len(ev.magnitudes):
            assert ev.preferred_magnitude() == ev.magnitudes[-1]
        if len(cat[0].focal_mechanisms):
            assert ev.preferred_focal_mechanism() == ev.focal_mechanisms[-1]

    def test_cat1_preferred_cache_empty(self, cat1_preferred_cache_empty):
        """ensure preferred still point to correct (not last) origins/mags"""
        cat = cat1_preferred_cache_empty
        validate_catalog(cat)
        self.preferred_ids_are_set(cat)
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
        """make sure a bad pick_id in arrivals raises assertion error"""
        with pytest.raises(AssertionError):
            validate_catalog(cat1_bad_arrival_pick_id)

    def test_duplicate_objects_raise(self, cat1_multiple_resource_ids):
        """
        Make sure an assertion error is raised on cat2 as it's resource
        ids are not unique.
        """
        with pytest.raises(AssertionError):
            validate_catalog(cat1_multiple_resource_ids)

    def test_empty_phase_hint_raises(self, cat1_no_pick_phase_hints):
        """ensure raises if any phase hints are undefined"""
        with pytest.raises(AssertionError):
            validate_catalog(cat1_no_pick_phase_hints)

    def test_empty_pick_wid_raises(self, cat1_no_pick_waveform_id):
        """ensure raise if any waveform ids are empty on picks"""
        with pytest.raises(AssertionError):
            validate_catalog(cat1_no_pick_waveform_id)

    def test_none_in_arrival_pick_id_fails(self, cat1_none_arrival_pick_id):
        """make sure if an arrival has a None pick validate raises"""
        with pytest.raises(AssertionError):
            validate_catalog(cat1_none_arrival_pick_id)

    def test_works_with_event(self, cat1):
        """ensure the method can also be called on an event"""
        validate_catalog(cat1[0])

    def test_duplicate_picks(self, cat1):
        """ensure raise if there are more than one p or s pick per station"""
        cat = cat1.copy()
        # Duplicating p pick at picks[2]
        pick = cat[0].picks[2]
        pick.time = pick.time + 60
        cat[0].picks.append(pick)
        with pytest.raises(AssertionError):
            obsplus.events.validate.check_duplicate_picks(cat)

    def test_s_before_p(self, cat1):
        """ensure raise if any s picks are before p picks"""
        cat = cat1.copy()
        # Set s time before p time
        # pick[3] is a s pick and pick[2] is a p pick
        cat[0].picks[3].time = cat[0].picks[2].time - 60
        with pytest.raises(AssertionError):
            obsplus.events.validate.check_pick_order(cat)

    def test_nullish_codes_replaced(self, cat_nullish_nslc_codes):
        """Nullish location codes should be replace with empty strings."""
        kwargs = dict(obj=cat_nullish_nslc_codes, cls=WaveformStreamID)
        for obj, _, _ in yield_obj_parent_attr(**kwargs):
            assert obj.location_code == ""

    def test_iaml_before_p(self, cat1):
        """
        ensure raise if there are any iaml picks that are before p picks
        """
        cat = cat1.copy()
        # Moving a iaml time before p time on the same station
        # picks[23] is a known iaml pick
        # picks[2] is a known p pick
        cat[0].picks[23].time = cat[0].picks[2].time - 60
        with pytest.raises(AssertionError):
            validate_catalog(cat)

    def test_p_lims(self, cat1):
        """ensure raise if there are any outlying P picks"""
        cat = cat1.copy()
        # Assigning p pick an outlying time (1 hour off)
        # picks[2] is a known p pick
        cat[0].picks[2].time = cat[0].picks[2].time + 60 * 60
        with pytest.raises(AssertionError):
            validate_catalog(cat, p_lim=30 * 60)

    def test_amp_lims(self, cat1):
        """ensure raise if there are any above limit amplitudes picks"""
        cat = cat1.copy()
        # Assigning an amplitude an above limit value
        cat[0].amplitudes[0].generic_amplitude = 1
        with pytest.raises(AssertionError):
            validate_catalog(cat, amp_lim=0.5)

    def test_amp_filts(self, cat1):
        """ensure raise if unexpected filter used"""
        cat = cat1.copy()
        amp = cat[0].amplitudes[0]
        # Assigning bad filter to an amplitude
        good_filt = "smi:local/Wood_Anderson_Simulation"
        bad_filt = "smi:local/Sean_Anderson_Simulation"
        rid = ResourceIdentifier(bad_filt, referred_object=amp)
        amp.filter_id = rid
        with pytest.raises(AssertionError):
            validate_catalog(cat, filter_ids=good_filt)

    def test_z_amps(self, cat1):
        """Raise if there are any amplitude picks on Z axis"""
        cat = cat1.copy()
        # Assigning iaml pick to a z channel
        # picks[23] is a known iaml pick
        cat[0].picks[23].waveform_id.channel_code = "HHZ"
        with pytest.raises(AssertionError):
            validate_catalog(cat, no_z_amps=True)

    def test_amp_times(self, cat1):
        """
        ensure raise if there are any amplitude times that don't match it's
        referred pick time
        """
        cat = cat1.copy()
        # Assigning bad time window to an amplitude
        pick = cat[0].amplitudes[0].pick_id.get_referred_object()
        tw = TimeWindow(begin=0, end=0.5, reference=pick.time + 10)
        cat[0].amplitudes[0].time_window = tw
        with pytest.raises(AssertionError):
            validate_catalog(cat)

    def test_duplicate_picks_ok_if_rejected(self, cat1):
        """
        Rejected picks should not count against duplicated
        """
        cat = cat1.copy()
        # get first non-rejected pick
        for pick in cat1[0].picks:
            if pick.evaluation_status != "rejected":
                pick = pick.copy()
                break
        else:
            raise ValueError("all picks rejected")
        pick.resource_id = obspy.core.event.ResourceIdentifier(referred_object=pick)
        pick.evaluation_status = "rejected"
        cat[0].picks.append(pick)
        # this should not raise
        validate_catalog(cat)

    def test_duplicate_station_different_network(self, cat1):
        """
        Ensure picks can have duplicated station codes if they have different
        network codes. See issue #173.
        """
        # Add a copy of first pick, add new resource id and a new network code
        new_pick1 = copy.deepcopy(cat1[0].picks[0])
        new_pick1.waveform_id.network_code = "NW"
        new_pick1.resource_id = ev.ResourceIdentifier()
        cat1[0].picks.append(new_pick1)
        # Do the same for network codes
        new_pick2 = copy.deepcopy(cat1[0].picks[0])
        new_pick2.waveform_id.location_code = "04"
        new_pick2.resource_id = ev.ResourceIdentifier()
        # test passes if this doesnt raise
        validate_catalog(cat1)
