"""
Tests for merging catalogs together.
"""
from glob import glob
from os.path import join

import numpy as np
import obspy
import obspy.core.event as ev
import pytest

from obsplus import validate_catalog
from obsplus.events.merge import merge_events, attach_new_origin, associate_merge
from obsplus.utils import yield_obj_parent_attr

CAT = obspy.read_events()
ORIGINS = [ori for eve in CAT for ori in eve.origins]
MAGNITUDES = [mag for eve in CAT for mag in eve.magnitudes]
ALLSORTS = ORIGINS + MAGNITUDES


# -------------------------- helper functions ----------------------- #


def extract_merge_catalogs(merge_directory):
    """ given a directory with two qmls, read in the qmls and return """
    files = glob(join(merge_directory, "*"))
    cat_path1 = [x for x in files if x.endswith("1.xml")]
    cat_path2 = [x for x in files if x.endswith("2.xml")]
    cat1 = obspy.read_events(cat_path1[0])
    cat2 = obspy.read_events(cat_path2[0])
    validate_catalog(cat1)
    validate_catalog(cat2)
    return cat1.copy(), cat2.copy()


# --------------------------- module fixtures ----------------------- #


@pytest.fixture(scope="function")
def merge_catalogs_function(qml_to_merge_paths):
    """ return a pair of catalogs for merge testing"""
    return extract_merge_catalogs(qml_to_merge_paths)


@pytest.fixture(scope="class")
def merge_catalog_basic(qml_to_merge_basic):
    """ return just the basic events used to test merging"""
    return extract_merge_catalogs(qml_to_merge_basic)


# ------------------------------- tests ------------------------------ #


class TestMergePicks:
    """
    Test that the picks and amplitudes of two catalogs can be cleanly merged
    together on the specified level.
    """

    # fixtures
    @pytest.fixture(scope="class")
    def two_catalogs(self, merge_catalog_basic):
        """
        Read the test seisan file twice, add .1 seconds to each of the
        picks and add some to generic amplitudes then return both catalogs.
        """
        cat1, cat2 = merge_catalog_basic
        for pick in cat2[0].picks:
            pick.time += 0.1
        for amplitude in cat2[0].amplitudes:
            amplitude.generic_amplitude *= 1.1
            amplitude.period *= 1.1
        return cat1.copy(), cat2.copy()

    @pytest.fixture(scope="class")
    def pick_resource_ids(self, merge_catalog_basic):
        """ return list of pick ids from the first events """
        cat1, _ = merge_catalog_basic
        pick_ids = [x.resource_id.id for x in cat1[0].picks]
        return pick_ids

    @pytest.fixture(scope="class")
    def amplitude_resource_ids(self, merge_catalog_basic):
        """ return list of pick ids from the first events """
        cat1, _ = merge_catalog_basic
        amplitude_ids = [x.resource_id.id for x in cat1[0].amplitudes]
        return amplitude_ids

    @pytest.fixture()
    def merged_catalogs(self, two_catalogs):
        """Merge two catalogs together."""
        cat1, cat2 = two_catalogs
        merge_events(cat1[0], cat2[0])
        return cat1, cat2

    @pytest.fixture()
    def merge_catalogs_delete_pick(self, qml_to_merge_basic):
        """ Delete a pick and amplitude from the new cat, merge with old."""
        cat1, cat2 = extract_merge_catalogs(qml_to_merge_basic)
        cat2[0].picks.pop(0)
        cat2[0].amplitudes.pop(0)
        merge_events(cat1[0], cat2[0])
        return cat1, cat2

    @pytest.fixture()
    def merge_catalogs_add_pick(self, qml_to_merge_basic):
        """ add a pick and amplitude to the new cat_name, merge with old """
        cat1, cat2 = extract_merge_catalogs(qml_to_merge_basic)
        # add new pick
        pick1 = cat2[0].picks[0]
        time = obspy.UTCDateTime.now()
        new_pick = obspy.core.event.Pick(
            time, waveform_id=pick1.waveform_id, phase_hint="S"
        )
        cat2[0].picks.append(new_pick)
        # add new amplitude
        amp = obspy.core.event.Amplitude(pick_id=new_pick.resource_id)
        cat2[0].amplitudes.append(amp)
        merge_events(cat1[0], cat2[0])
        return cat1, cat2

    @pytest.fixture()
    def merge_catalogs_add_bad_amplitude(self, qml_to_merge_basic):
        """
        Add an amplitude that has no pick reference. It should not get
        merged into the events.
        """
        cat1, cat2 = extract_merge_catalogs(qml_to_merge_basic)
        # add new amplitude
        amp = obspy.core.event.Amplitude()
        cat2[0].amplitudes.append(amp)
        merge_events(cat1[0], cat2[0])
        return cat1, cat2

    # tests
    def test_pick_times(self, merged_catalogs):
        """ test that the times are the same in the picks """
        cat1, cat2 = merged_catalogs
        assert len(cat1[0].picks) == len(cat2[0].picks)
        for pick1, pick2 in zip(cat1[0].picks, cat2[0].picks):
            assert pick1.time == pick2.time
            assert pick1.resource_id != pick2.resource_id
        validate_catalog(cat1)
        validate_catalog(cat2)

    def test_amplitudes(self, merged_catalogs):
        """ ensure the amplitudes are the same """
        cat1, cat2 = merged_catalogs
        assert len(cat1[0].amplitudes) == len(cat2[0].amplitudes)
        for amp1, amp2 in zip(cat1[0].amplitudes, cat2[0].amplitudes):
            assert amp1.generic_amplitude == amp2.generic_amplitude
            assert amp1.period == amp2.period
            assert amp1.resource_id != amp2.resource_id

    def test_cat1_pick_ids_unchanged(self, merged_catalogs, pick_resource_ids):
        """
        Ensure the resource_ids of the first events are unchanged despite
        merge.
        """
        cat1, _ = merged_catalogs
        new_pick_ids = [x.resource_id.id for x in cat1[0].picks]
        assert new_pick_ids == pick_resource_ids

    def test_cat1_amplitude_ids_unchanged(
        self, merged_catalogs, amplitude_resource_ids
    ):
        """ ensure the resource IDs on the amplitudes havent changed """
        cat1, _ = merged_catalogs
        new_amplitude_ids = [x.resource_id.id for x in cat1[0].amplitudes]
        assert amplitude_resource_ids == new_amplitude_ids

    def test_deleted_pick(self, merge_catalogs_delete_pick):
        """
        Test that when a pick is deleted from the new cat_name and the
        CATALOG_PATH are merged it is also deleted from the old.
        """
        cat1, cat2 = merge_catalogs_delete_pick
        assert len(cat1[0].picks) == len(cat2[0].picks)
        for pick1, pick2 in zip(cat1[0].picks, cat2[0].picks):
            assert pick1.time == pick2.time
        assert len(cat1[0].amplitudes) == len(cat2[0].amplitudes)
        for amp1, amp2 in zip(cat1[0].amplitudes, cat2[0].amplitudes):
            assert amp1.generic_amplitude == amp2.generic_amplitude

    def test_add_pick(self, merge_catalogs_add_pick):
        """
        Test that when a pick is added to the new cat_name it shows up
        in the old cat_name.
        """
        cat1, cat2 = merge_catalogs_add_pick
        assert len(cat1[0].picks) == len(cat2[0].picks)
        for pick1, pick2 in zip(cat1[0].picks, cat2[0].picks):
            assert pick1.time == pick2.time
        assert len(cat1[0].amplitudes) == len(cat2[0].amplitudes)
        for amp1, amp2 in zip(cat1[0].amplitudes, cat2[0].amplitudes):
            assert amp1.generic_amplitude == amp2.generic_amplitude
        validate_catalog(cat1)
        validate_catalog(cat2)

    def test_bad_amplitude(self, merge_catalogs_add_bad_amplitude):
        """ ensure an amplitude with no pick id doesnt get merged """
        cat1, cat2 = merge_catalogs_add_bad_amplitude
        assert len(cat1[0].amplitudes) == (len(cat2[0].amplitudes) - 1)


class TestAttachNewOrigin:
    """ tests for attaching a new origin to a events """

    # functions
    def origin_is_preferred(self, cat, origin):
        """Ensure the origin is preferred."""
        por = cat[0].preferred_origin()
        assert por is not None
        assert por == origin

    def ensure_common_arrivals(self, ori1, ori2):
        """Ensure origin is a modified version of cat2's first origin."""
        origin_pick_dict = {x.pick_id.id for x in ori1.arrivals}
        cat2_pick_dict = {x.pick_id.id for x in ori2.arrivals}
        assert set(origin_pick_dict) == set(cat2_pick_dict)

    # fixtures
    @pytest.fixture(scope="function")
    def origin_pack(self, merge_catalogs_function):
        """
        Using the second events from the merge set, create an origin
        object that will be attached to the first events.
        """
        cat1 = merge_catalogs_function[0].copy()
        cat2 = merge_catalogs_function[1].copy()
        origin = cat2[0].preferred_origin() or cat2[0].origins[-1]
        origin.time += 1.0
        origin.latitude *= 1.05
        origin.longitude *= 1.05
        rid = obspy.core.event.ResourceIdentifier()
        rid.set_referred_object(origin)
        origin.resource_id = rid
        # ensure all picks are accounted for on second events
        origin_pick_id = {x.pick_id.id for x in origin.arrivals}
        pick_ids = {x.resource_id.id for x in cat2[0].picks}
        assert origin_pick_id.issubset(pick_ids)
        # ensure there is nothing invalid about the catalogs
        validate_catalog(cat1)
        validate_catalog(cat2)

        return cat1, cat2, origin

    @pytest.fixture(scope="function")
    def append_origin_catalog(self, origin_pack):
        """ attach the new origin to the first event without index """
        cat1, cat2, origin = origin_pack
        self.ensure_common_arrivals(origin, cat2[0].origins[0])
        attach_new_origin(cat1[0], cat2[0], origin, preferred=True)
        return cat1

    @pytest.fixture(scope="function")
    def insert_origin_catalog(self, origin_pack):
        """ insert the origin to overwrite old origin """
        cat1, cat2, origin = origin_pack
        # ensure origin is a modified version of cat2's first origin
        self.ensure_common_arrivals(origin, cat2[0].origins[0])
        attach_new_origin(cat1[0], cat2[0], origin, preferred=True, index=0)
        validate_catalog(cat1)
        return cat1

    # tests
    def test_append_origin(self, append_origin_catalog, origin_pack):
        """Ensure the origins was attached and is preferred and comes last."""
        _, _, origin = origin_pack
        self.origin_is_preferred(append_origin_catalog, origin)
        assert origin == append_origin_catalog[0].origins[-1]

    def test_insert_origins(self, insert_origin_catalog, origin_pack):
        """ ensure the origin is now the first in the list """
        _, _, origin = origin_pack

        self.origin_is_preferred(insert_origin_catalog, origin)
        assert origin == insert_origin_catalog[0].origins[0]

    def test_insert_origin_out_of_bounds(self, origin_pack):
        """ ensure the origin is still found even if bogus index was used """
        cat1, cat2, origin = origin_pack

        # ensure origin is a modified version of cat2's first origin
        self.ensure_common_arrivals(origin, cat2[0].origins[0])

        with pytest.warns(UserWarning) as w:
            attach_new_origin(cat1[0], cat2[0], origin, preferred=True, index=500)

        assert len(w) != 0

        validate_catalog(cat1)

        self.origin_is_preferred(cat1, origin)
        assert origin in cat1[0].origins
        assert cat1[0].origins[-1] == origin


class TestMergeNewPicks:
    """Tests for merging new picks into old catalogs."""

    @pytest.fixture()
    def merge_base_event(self, bingham_catalog):
        """The base event for merging."""
        event = bingham_catalog[0].copy()
        return event

    @pytest.fixture()
    def simple_catalog_to_merge(self, bingham_catalog):
        """
        Create a simple catalog to merge into bingham_cat using only one event.
        """
        cat = obspy.Catalog(events=bingham_catalog[:2]).copy()
        # drop first pick
        cat[0].picks = cat[0].picks[1:]
        # modify the picks to whole seconds, reset pick IDS
        for pick, _, _ in yield_obj_parent_attr(cat, ev.Pick):
            pick.time -= (pick.time.timestamp) % 1
            pick.id = ev.ResourceIdentifier(referred_object=pick)
        return cat

    @pytest.fixture()
    def miss_merge_catalog(self, bingham_catalog):
        """
        Create a catalog whose events are too far in time to be merged.
        """
        cat = obspy.Catalog(events=bingham_catalog[2:]).copy()
        return cat

    def test_picks_merged(self, simple_catalog_to_merge, merge_base_event):
        """Test merging. """
        merged = associate_merge(merge_base_event, simple_catalog_to_merge)
        # iterate and test each pick
        first_pick = merged.picks[0]
        assert (first_pick).time.timestamp % 1 != 0
        for pick in merged.picks[1:]:
            remain = pick.time.timestamp % 1
            close_to_1 = np.isclose(remain, 1, atol=1e-05)
            close_to_0 = np.isclose(remain, 0, atol=1e-05)
            assert close_to_0 or close_to_1

    def test_miss_merge(self, merge_base_event, miss_merge_catalog):
        """Ensure a catalog which is far away from the base does nothing."""
        original = merge_base_event.copy()
        cat = associate_merge(merge_base_event, miss_merge_catalog)
        assert cat == original
