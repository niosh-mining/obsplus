"""
Tests for utils for events submodule
"""

import copy
import time
from pathlib import Path

import numpy as np
import obsplus
import obsplus.utils.events
import obspy
import obspy.core.event as ev
import pytest
from obsplus import get_preferred
from obsplus.events import validate
from obsplus.exceptions import ValidationError
from obsplus.interfaces import EventClient
from obsplus.utils.events import (
    bump_creation_version,
    duplicate_events,
    get_event_client,
    make_origins,
    prune_events,
    strip_events,
)
from obsplus.utils.misc import get_instances_from_tree
from obspy.core.event import ResourceIdentifier

CAT = obspy.read_events()


class TestDuplicateEvent:
    """Tests for duplicating events."""

    @pytest.fixture
    def catalog(self):
        """Return default events"""
        return obspy.read_events()

    @pytest.fixture
    def duplicated_catalog(self, catalog):
        """Return an event duplicated from first"""
        return duplicate_events(catalog)

    @pytest.fixture
    def duplicated_big_catalog(self, catalog_cache):
        """Duplicate the big catalog."""
        return obsplus.duplicate_events(catalog_cache["cat6"])

    def test_return_type(self, duplicated_catalog):
        """Ensure a events was returned"""
        assert isinstance(duplicated_catalog, obspy.Catalog)

    def test_unique_resource_ids(self, catalog, duplicated_catalog):
        """Ensure all resource ids are unique in duplicated event"""
        ev1, ev2 = catalog, duplicated_catalog
        rids1 = {x for x in get_instances_from_tree(ev1, cls=ResourceIdentifier)}
        rids2 = {x for x in get_instances_from_tree(ev2, cls=ResourceIdentifier)}
        assert len(rids1) and len(rids2)  # ensure rids not empty
        commons = rids1 & rids2
        # all shared resource_ids should not refer to an object
        assert all(x.get_referred_object() is None for x in commons)

    def test_duplicated(self, catalog, duplicated_catalog):
        """Ensure duplicated is equal on all aspects except resource id"""
        cat1, cat2 = catalog, duplicated_catalog
        origin_attrs = ("latitude", "longitude", "depth", "time")

        assert len(cat1) == len(cat2)
        for ev1, ev2 in zip(cat1, cat2):
            or1 = ev1.preferred_origin() or ev1.origins[-1]
            or2 = ev2.preferred_origin() or ev2.origins[-1]
            for origin_attr in origin_attrs:
                assert getattr(or1, origin_attr) == getattr(or2, origin_attr)

    def test_duplicated_catalog_valid(self, duplicated_big_catalog):
        """Ensure the duplicated events is valid"""
        obsplus.validate_catalog(duplicated_big_catalog)

    def test_interconnected_rids(self, catalog_cache):
        """Tests for ensuring resource IDs are changed to point to new
        objects. This can get messed up when there are many objects
        that point to resource ids of other objects. EG picks/amplitudes.
        """
        cat = duplicate_events(catalog_cache["interconnected"])
        # create a dict of pick ids and others that refer to picks
        pids = {str(x.resource_id): x for x in cat[0].picks}
        assert len(cat[0].origins) == 1, "there should be one origin"
        origin = cat[0].origins[0]
        apids = {str(x.pick_id): x for x in origin.arrivals}
        ampids = {str(x.pick_id): x for x in cat[0].amplitudes}
        assert set(apids).issubset(pids)
        assert set(ampids).issubset(pids)
        validate.validate_catalog(cat)


class TestPruneEvents:
    """Tests for removing unused and rejected objects from events."""

    @pytest.fixture
    def event_rejected_pick(self):
        """Create an event with a rejected pick."""
        wid = ev.WaveformStreamID(seed_string="UU.TMU.01.ENZ")
        time = obspy.UTCDateTime("2019-01-01")
        pick1 = ev.Pick(
            time=time, waveform_id=wid, phase_hint="P", evaluation_status="rejected"
        )
        pick2 = ev.Pick(time=time, waveform_id=wid, phase_hint="P")
        return ev.Event(picks=[pick1, pick2])

    @pytest.fixture
    def event_non_orphaned_rejected_pick(self, event_rejected_pick):
        """Change both picks to rejected but reference one from arrival."""
        eve = event_rejected_pick.copy()
        picks = eve.picks
        for pick in picks:
            pick.evaluation_status = "rejected"
        first_pick_rid = str(picks[0].resource_id)
        # now create origin and arrival
        arrival = ev.Arrival(pick_id=first_pick_rid)
        origin = ev.Origin(
            latitude=0, longitude=0, depth=10, time=picks[0].time, arrivals=[arrival]
        )
        eve.origins.append(origin)
        return eve

    @pytest.fixture
    def catalog_rejected_orphan_origin(self):
        """Create a catalog with an orphaned rejected origin."""
        cat = obspy.read_events()
        origin = cat[0].origins[0]
        origin.arrivals.clear()
        origin.evaluation_status = "rejected"
        cat[0].magnitudes.clear()
        return cat

    def test_copy_made(self):
        """Prune events should make a copy, not modify the original."""
        cat = obspy.read_events()
        assert prune_events(cat) is not cat

    def test_pick_gone(self, event_rejected_pick):
        """Ensure the pick was removed."""
        picks = prune_events(event_rejected_pick)[0].picks
        assert all([x.evaluation_status != "rejected" for x in picks])

    def test_non_orphan_rejected_kept(self, event_non_orphaned_rejected_pick):
        """Ensure rejected things are kept if their parents are not rejected."""
        ev = prune_events(event_non_orphaned_rejected_pick)[0]
        # One pick gets removed, the other is kept
        assert len(ev.picks) == 1
        assert ev.picks[0].evaluation_status == "rejected"
        por = obsplus.utils.events.get_preferred(ev, "origin")
        assert len(por.arrivals) == 1

    def test_one_rejected_origin(self, catalog_rejected_orphan_origin):
        """Ensure a rejected origin, with no other references, is removed."""
        event = catalog_rejected_orphan_origin[0]
        origin_count_before = len(event.origins)
        out = obsplus.utils.events.prune_events(event)
        assert len(out[0].origins) < origin_count_before


class TestStripEvents:
    """Tests for stripping off derivative and non-reviewed data"""

    # Fixtures
    @pytest.fixture
    def empty_cat(self):
        """Return an empty catalog"""
        return ev.Catalog()

    @pytest.fixture
    def cat_w_two_events(self, empty_cat):
        """Return a catalog with two empty events"""
        empty_cat.append(ev.Event())
        empty_cat.append(ev.Event())
        return empty_cat

    @pytest.fixture
    def cat_picks(self, cat_w_two_events):
        """Add some picks to the events, including rejected picks"""
        eve = cat_w_two_events[0]
        wid = ev.WaveformStreamID(seed_string="UU.TMU.01.ENZ")
        time = obspy.UTCDateTime()
        eve.picks.append(
            ev.Pick(
                time=time, waveform_id=wid, phase_hint="P", evaluation_status="reviewed"
            )
        )
        eve.picks.append(
            ev.Pick(
                time=time, waveform_id=wid, phase_hint="P", evaluation_status="rejected"
            )
        )
        eve = cat_w_two_events[1]
        eve.picks.append(
            ev.Pick(
                time=time,
                waveform_id=wid,
                phase_hint="P",
                evaluation_status="preliminary",
            )
        )
        eve.picks.append(
            ev.Pick(
                time=time,
                waveform_id=wid,
                phase_hint="P",
                evaluation_status="confirmed",
            )
        )
        eve.picks.append(
            ev.Pick(
                time=time, waveform_id=wid, phase_hint="P", evaluation_status="final"
            )
        )
        return cat_w_two_events

    @pytest.fixture
    def event_amplitudes(self, cat_picks):
        """Return an event with some amplitudes"""
        eve = cat_picks[0]
        eve.amplitudes.append(
            ev.Amplitude(generic_amplitude=1, evaluation_status="reviewed")
        )
        eve.amplitudes.append(
            ev.Amplitude(generic_amplitude=1, evaluation_status="rejected")
        )
        return eve

    @pytest.fixture
    def event_linked_amplitudes(self, event_amplitudes):
        """Link all of the amplitudes to a rejected pick"""
        for amp in event_amplitudes.amplitudes:
            amp.pick_id = event_amplitudes.picks[1].resource_id
        return event_amplitudes

    @pytest.fixture
    def event_description(self, event_amplitudes):
        """Add some event descriptions to the event"""
        event_amplitudes.event_descriptions.append(ev.EventDescription(text="Keep Me"))
        event_amplitudes.event_descriptions.append(ev.EventDescription(text="Toss Me"))
        return event_amplitudes

    @pytest.fixture
    def event_origins(self, event_description):
        """Add an origin to the event"""
        event_description.origins.append(
            ev.Origin(time=obspy.UTCDateTime(), longitude=-111, latitude=37)
        )
        return event_description

    @pytest.fixture
    def event_focal_mech(self, event_origins):
        """Add a focal mechanism to the event"""
        event_origins.focal_mechanisms.append(ev.FocalMechanism())
        return event_origins

    @pytest.fixture
    def event_station_mags(self, event_focal_mech):
        """Add a focal mechanism to the event"""
        event_focal_mech.station_magnitudes.append(ev.StationMagnitude())
        return event_focal_mech

    @pytest.fixture
    def event_magnitudes(self, event_station_mags):
        """Add a focal mechanism to the event"""
        event_station_mags.magnitudes.append(ev.Magnitude())
        return event_station_mags

    # Tests
    def test_empty(self, empty_cat):
        """Make sure an empty catalog can pass through"""
        strip_events(empty_cat)

    def test_empty_events(self, cat_w_two_events):
        """Make sure empty events can pass through"""
        out = strip_events(cat_w_two_events)
        assert len(out) == 2

    def test_copy_made(self):
        """Prune events should make a copy, not modify the original."""
        cat = obspy.read_events()
        assert strip_events(cat) is not cat

    def test_acceptable_eval_statuses(self, cat_picks):
        """
        Make sure only picks with acceptable evaluation statuses get through
        (not rejected or preliminary)
        """
        out = strip_events(cat_picks)
        assert len(out[0].picks) == 1
        assert len(out[1].picks) == 3

    def test_custom_acceptable_eval_statuses(self, cat_picks):
        """
        Make sure the user can specify the evaluation status of stuff to reject
        """
        out = strip_events(
            cat_picks, reject_evaluation_status=["preliminary", "confirmed", "rejected"]
        )
        assert len(out[0].picks) == 1
        assert len(out[1].picks) == 1

    def test_only_reviewed_amplitudes(self, event_amplitudes):
        """Make sure only non-rejected amplitudes make it through"""
        out = strip_events(event_amplitudes)
        assert len(out.amplitudes) == 1

    def test_linked_amplitudes_picks(self, event_linked_amplitudes):
        """
        Make sure any amplitudes (rejected or otherwise) that are linked to a
        rejected pick get tossed
        """
        out = strip_events(event_linked_amplitudes)
        assert len(out.amplitudes) == 0

    def test_only_first_event_description(self, event_description):
        """Make sure only the first event description survives"""
        text = event_description.event_descriptions[0].text
        out = strip_events(event_description)
        assert len(out.event_descriptions) == 1
        assert out.event_descriptions[0].text == text

    def test_no_origins(self, event_origins):
        """
        Make sure there are no origins attached to the event and that the
        preferred origin doesn't refer to anything
        """
        out = strip_events(event_origins)
        assert len(out.origins) == 0
        assert out.preferred_origin() is None

    def test_no_focal_mechanisms(self, event_focal_mech):
        """
        Make sure there are no focal mechanisms attached to the event and that
        the preferred focal mechanism doesn't refer to anything
        """
        out = strip_events(event_focal_mech)
        assert len(out.focal_mechanisms) == 0
        assert out.preferred_focal_mechanism() is None

    def test_no_station_magnitudes(self, event_station_mags):
        """
        Make sure there are no station magnitudes attached to the event
        """
        out = strip_events(event_station_mags)
        assert len(out.station_magnitudes) == 0

    def test_no_magnitudes(self, event_magnitudes):
        """
        Make sure there are no magnitudes attached to the event and that the
        preferred magnitude doesn't refer to anything
        """
        out = strip_events(event_magnitudes)
        assert len(out.magnitudes) == 0
        assert out.preferred_magnitude() is None

    def test_not_a_catalog(self):
        """Addresses #276"""
        with pytest.raises(TypeError, match="Catalog"):
            strip_events("abcdefg")


class TestBumpCreationVersion:
    """tests for the bump_creation_version function"""

    # fixtures
    @pytest.fixture(scope="class")
    def cat(self):
        """A basic obspy cat_name"""
        return CAT.copy()

    @pytest.fixture(scope="class")
    def eve_cis(self, cat):
        """Return the original version and the bumped version"""
        ev1 = cat[0].origins[0]
        cl1 = copy.deepcopy(ev1.creation_info)
        ev2 = cat[0].origins[0].copy()
        bump_creation_version(ev2)
        cl2 = ev2.creation_info
        return cl1, cl2

    @pytest.fixture(scope="class")
    def multi_version(self, cat):
        """
        Bump version, copy event, bump version again, return creation info.
        """
        eve1 = cat[0]
        bump_creation_version(eve1)
        eve2 = eve1.copy()
        time.sleep(0.001)
        bump_creation_version(eve2)
        return eve1.creation_info, eve2.creation_info

    @pytest.fixture(scope="class")
    def int_version(self, cat):
        """
        Bump version, copy event, bump version again, set version to int,
        bump again.
        """
        eve1 = cat[0]
        bump_creation_version(eve1)
        eve1.creation_info.version = 0
        bump_creation_version(eve1)
        return eve1.creation_info

    # tests
    def test_bump_version(self, eve_cis):
        """Test that the version gets bumped once on default cat_name"""
        ci1, ci2 = eve_cis
        ct1, ct2 = ci1.creation_time, ci2.creation_time
        assert isinstance(ct2, obspy.UTCDateTime)
        if isinstance(ct1, obspy.UTCDateTime):
            assert ct1 < ct2
        assert ci2.version is not None

    def test_bump_twice(self, multi_version):
        """Test that the version can be bumped twice"""
        ci1, ci2 = multi_version
        ct1, ct2 = ci1.creation_time, ci2.creation_time
        v1, v2 = ci1.version, ci2.version
        for update_time in [ct1, ct2]:
            assert isinstance(update_time, obspy.UTCDateTime)
        for ver in [v1, v2]:
            assert isinstance(ver, str | int)
        assert ct2 > ct1
        assert v2 > v1

    def test_bump_int_version(self, int_version):
        """Ensure bumping an integer version can happen"""
        assert int_version.version == "1"

    def test_bump_version_on_bad_object(self):
        """
        Ensure bumping the version on a non-obspy object doesnt error
        and doesnt add creation_info.
        """
        test_obj = "some_string"
        bump_creation_version(test_obj)
        assert not hasattr(test_obj, "creation_info")


class TestGetPreferred:
    """
    Tests for getting preferred things form events.
    """

    def test_events_no_preferred(self):
        """Test that the last origin gets returned."""
        event = obspy.read_events()[0]
        event.preferred_origin_id = None  # clear origin_id
        assert event.origins[-1] == get_preferred(event, "origin")

    def test_preferred_no_origins(self):
        """
        When the preferred id is set but origin is empty None should be
        returned.
        """
        event = obspy.read_events()[0]
        # clear origins and ensure resource_id is not holding a reference
        event.origins.clear()
        rid = str(ev.ResourceIdentifier())
        event.preferred_origin_id = rid
        # It should now return None
        with pytest.warns(UserWarning):
            assert get_preferred(event, "origin") is None
        # but if init_empty it should return an empty origin
        with pytest.warns(UserWarning):
            ori = get_preferred(event, "origin", init_empty=True)
        assert isinstance(ori, ev.Origin)

    def test_bad_preferred_origin(self):
        """Ensure the bad preferred just returns last in list"""
        eve = obspy.read_events()[0]
        eve.preferred_origin_id = "bob"
        with pytest.warns(UserWarning) as w:
            preferred_origin = get_preferred(eve, "origin")
        assert len(w) == 1
        assert preferred_origin is eve.origins[-1]


class TestMakeOrigins:
    """Tests for the ensure origin function."""

    @pytest.fixture(scope="class")
    def inv(self):
        """Return the crandall inventory."""
        ds = obsplus.load_dataset("crandall_test")
        return ds.station_client.get_stations()

    @pytest.fixture(scope="class")
    def cat_only_picks(self, crandall_dataset):
        """Return a catalog with only picks, no origins or magnitudes"""
        cat = crandall_dataset.event_client.get_events().copy()
        for event in cat:
            event.preferred_origin_id = None
            event.preferred_magnitude_id = None
            event.origins.clear()
            event.magnitudes.clear()
        return cat

    @pytest.fixture(scope="class")
    def cat_bad_first_picks(self, cat_only_picks):
        """Return a catalog with only picks, no origins or magnitudes"""
        # change the first picks to a station not in the inventory
        cat = cat_only_picks.copy()
        bad_wid = ev.WaveformStreamID(seed_string="SM.RDD..HHZ")
        for event in cat:
            first_pick = sorted(event.picks, key=lambda x: x.time)[0]
            first_pick.waveform_id = bad_wid
            event.origins.clear()
        return cat

    @pytest.fixture(scope="class")
    def cat_added_origins(self, cat_only_picks, inv):
        """Run make_origins on the catalog with only picks and return"""
        # get corresponding inventory
        return make_origins(events=cat_only_picks, inventory=inv)

    @pytest.fixture(scope="class")
    def strange_picks_added_origins(self, inv):
        """Make sure "rejected" picks and oddball phase hints get skipped"""
        # Pick w/ good phase hint but bad evaluation status
        pick1 = ev.Pick(
            time=obspy.UTCDateTime(),
            phase_hint="P",
            evaluation_status="rejected",
            waveform_id=ev.WaveformStreamID(seed_string="UU.TMU..HHZ"),
        )
        # Pick w/ bad phase hint but good evaluation status
        pick2 = ev.Pick(
            time=obspy.UTCDateTime(),
            phase_hint="junk",
            evaluation_status="reviewed",
            waveform_id=ev.WaveformStreamID(seed_string="UU.CTU..HHZ"),
        )
        # Pick w/ good phase hint and evaluation status
        pick3 = ev.Pick(
            time=obspy.UTCDateTime(),
            phase_hint="S",
            waveform_id=ev.WaveformStreamID(seed_string="UU.SRU..HHN"),
        )
        eve = ev.Event()
        eve.picks = [pick1, pick2, pick3]
        return make_origins(events=eve, inventory=inv, phase_hints=["P", "S"]), pick3

    def test_all_events_have_origins(self, cat_added_origins):
        """Ensure all the events do indeed have origins"""
        for event in cat_added_origins:
            assert event.origins, f"{event} has no origins"

    def test_origins_have_time_and_location(self, cat_added_origins):
        """All added origins should have both times and locations."""
        for event in cat_added_origins:
            for origin in event.origins:
                assert isinstance(origin.time, obspy.UTCDateTime)
                assert origin.latitude is not None
                assert origin.longitude is not None
                assert origin.depth is not None

    def test_correct_origin_time(self, strange_picks_added_origins):
        """
        Ensure newly attached origin used the correct pick to get the location.
        """
        ori = strange_picks_added_origins[0].origins[0]
        time = strange_picks_added_origins[1].time
        t1, t2 = ori.time.timestamp, time.timestamp
        assert np.isclose(t1, t2)

    def test_bad_first_not_in_inventory(self, cat_bad_first_picks, inv):
        """Ensure function raises when bad first picks are found."""
        with pytest.raises(ValidationError):
            make_origins(cat_bad_first_picks, inv)

    def test_no_picks(self, inv):
        """Should raise if no picks are found."""
        cat = obspy.read_events()
        for event in cat:
            event.origins.clear()
        with pytest.raises(ValidationError):
            make_origins(cat, inv)


class TestGetSeedId:
    """Tests for the get_seed_id function"""

    def test_get_seed_id(self):
        """Make sure it is possible to retrieve the seed id"""
        wid = obspy.core.event.WaveformStreamID(
            network_code="AA",
            station_code="BBB",
            location_code="CC",
            channel_code="DDD",
        )
        pick = obspy.core.event.Pick(waveform_id=wid)
        amp = obspy.core.event.Amplitude(pick_id=pick.resource_id)
        station_mag = obspy.core.event.StationMagnitude(amplitude_id=amp.resource_id)
        station_mag_cont = obspy.core.event.StationMagnitudeContribution(
            station_magnitude_id=station_mag.resource_id
        )
        seed = obsplus.utils.events.get_seed_id(station_mag_cont)
        assert seed == "AA.BBB.CC.DDD"

    def test_no_seed_id(self):
        """Make sure raises AttributeError if no seed info found"""
        with pytest.raises(AssertionError):
            obsplus.utils.events.get_seed_id(obspy.core.event.Pick())

    def test_unsupported(self):
        """Make sure an unsupported object raises TypeError"""
        with pytest.raises(TypeError):
            obsplus.utils.events.get_seed_id(obspy.core.event.Origin())


class TestGetEventClient:
    """Tests for getting an event client from various sources."""

    def test_from_catalog(self):
        """Tests for getting client from a catalog."""
        cat = obspy.read_events()
        out = get_event_client(cat)
        assert cat == out
        assert isinstance(cat, EventClient)

    def test_from_event_bank(self, default_ebank):
        """Test getting events from an eventbank."""
        out = get_event_client(default_ebank)
        assert isinstance(out, EventClient)

    def test_from_directory(self, simple_event_dir):
        """Test getting events from a directory."""
        out = get_event_client(simple_event_dir)
        assert isinstance(out, EventClient)

    def test_from_file(self, simple_event_dir):
        """Test getting events from a file."""
        # get first file
        first = next(iter(Path(simple_event_dir).rglob("*.xml")))
        out = get_event_client(first)
        assert isinstance(out, EventClient)
        assert isinstance(out, obspy.Catalog)

    def test_from_event(self):
        """An event should be converted to a catalog."""
        event = obspy.read_events()[0]
        out = get_event_client(event)
        assert isinstance(out, EventClient)
