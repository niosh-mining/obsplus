"""
Tests for utils for events submodule
"""
import os
import warnings
from pathlib import Path

import obspy
import obspy.core.event as ev
import pytest
from obspy.core.event import Comment, ResourceIdentifier

import obsplus
import obsplus.events.utils
from obsplus.events import validate
from obsplus.events.utils import (
    bump_creation_version,
    get_event_path,
    duplicate_events,
    catalog_to_directory,
    make_origins,
    prune_events,
)
from obsplus.utils import get_instances
from obsplus import get_preferred

CAT = obspy.read_events()


class TestDuplicateEvent:
    @pytest.fixture
    def catalog(self):
        """ return default events """
        return obspy.read_events()

    @pytest.fixture
    def duplicated_catalog(self, catalog):
        """ return an event duplicated from first """
        return duplicate_events(catalog)

    @pytest.fixture
    def duplicated_big_catalog(self, catalog_cache):
        return obsplus.duplicate_events(catalog_cache["cat6"])

    def test_return_type(self, duplicated_catalog):
        """ ensure a events was returned """
        assert isinstance(duplicated_catalog, obspy.Catalog)

    def test_unique_resource_ids(self, catalog, duplicated_catalog):
        """ ensure all resource ids are unique in duplicated event """
        ev1, ev2 = catalog, duplicated_catalog
        rids1 = {x for x in get_instances(ev1, ResourceIdentifier)}
        rids2 = {x for x in get_instances(ev2, ResourceIdentifier)}
        assert len(rids1) and len(rids2)  # ensure rids not empty
        commons = rids1 & rids2
        # all shared resource_ids should not refer to an object
        assert all(x.get_referred_object() is None for x in commons)

    def test_duplicated(self, catalog, duplicated_catalog):
        """ ensure duplicated is equal on all aspects except resource id """
        cat1, cat2 = catalog, duplicated_catalog
        origin_attrs = ("latitude", "longitude", "depth", "time")

        assert len(cat1) == len(cat2)
        for ev1, ev2 in zip(cat1, cat2):
            or1 = ev1.preferred_origin() or ev1.origins[-1]
            or2 = ev2.preferred_origin() or ev2.origins[-1]
            for origin_attr in origin_attrs:
                assert getattr(or1, origin_attr) == getattr(or2, origin_attr)

    def test_duplicated_catalog_valid(self, duplicated_big_catalog):
        """ ensure the duplicated events is valid """
        obsplus.validate_catalog(duplicated_big_catalog)

    def test_interconnected_rids(self, catalog_cache):
        """ Tests for ensuring resource IDs are changed to point to new
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
    """ Tests for removing unused and rejected objects from events. """

    @pytest.fixture
    def event_rejected_pick(self):
        """ Create an event with a rejected pick. """
        wid = ev.WaveformStreamID(seed_string="UU.TMU.01.ENZ")
        time = obspy.UTCDateTime("2019-01-01")
        pick1 = ev.Pick(
            time=time, waveform_id=wid, phase_hint="P", evaluation_status="rejected"
        )
        pick2 = ev.Pick(time=time, waveform_id=wid, phase_hint="P")
        return ev.Event(picks=[pick1, pick2])

    @pytest.fixture
    def event_non_orphaned_rejected_pick(self, event_rejected_pick):
        """ Change both picks to rejected but reference one from arrival. """
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

    def test_copy_made(self):
        """ Prune events should make a copy, not modify the original. """
        cat = obspy.read_events()
        assert prune_events(cat) is not cat

    def test_pick_gone(self, event_rejected_pick):
        """ Ensure the pick was removed. """
        picks = prune_events(event_rejected_pick)[0].picks
        assert all([x.evaluation_status != "rejected" for x in picks])

    def test_non_orphan_rejected_kept(self, event_non_orphaned_rejected_pick):
        """ Ensure rejected things are kept if their parents are not rejected. """
        ev = prune_events(event_non_orphaned_rejected_pick)[0]
        # One pick gets removed, the other is kept
        assert len(ev.picks) == 1
        assert ev.picks[0].evaluation_status == "rejected"
        por = obsplus.events.utils.get_preferred(ev, "origin")
        assert len(por.arrivals) == 1


class TestBumpCreationVersion:
    """ tests for the bump_creation_version function """

    # fixtures
    @pytest.fixture(scope="class")
    def cat(self):
        """ A basic obspy cat_name """
        return CAT.copy()

    @pytest.fixture(scope="class")
    def eve_cis(self, cat):
        """ return the original version and the bumped version """
        ev1 = cat[0].origins[0]
        cl1 = ev1.creation_info
        ev2 = cat[0].origins[0].copy()
        bump_creation_version(ev2)
        cl2 = ev2.creation_info
        return cl1, cl2

    @pytest.fixture(scope="class")
    def multi_version(self, cat):
        """ bump version, copy event, bump version again,
        return creation info """
        eve1 = cat[0]
        bump_creation_version(eve1)
        eve2 = eve1.copy()
        bump_creation_version(eve2)
        return eve1.creation_info, eve2.creation_info

    @pytest.fixture(scope="class")
    def int_version(self, cat):
        """ bump version, copy event, bump version again,
        set version to int, bump again """
        eve1 = cat[0]
        bump_creation_version(eve1)
        eve1.creation_info.version = 0
        bump_creation_version(eve1)
        return eve1.creation_info

    # tests
    def test_bump_version(self, eve_cis):
        """ test that the version gets bumped once on default cat_name """
        ci1, ci2 = eve_cis
        ct1, ct2 = ci1.creation_time, ci2.creation_time
        assert isinstance(ct2, obspy.UTCDateTime)
        if isinstance(ct1, obspy.UTCDateTime):
            assert ct1 < ct2
        assert ci2.version is not None

    def test_bump_twice(self, multi_version):
        """ test that the version can be bumped twice """
        ci1, ci2 = multi_version
        ct1, ct2 = ci1.creation_time, ci2.creation_time
        v1, v2 = ci1.version, ci2.version
        for time in [ct1, ct2]:
            assert isinstance(time, obspy.UTCDateTime)
        for ver in [v1, v2]:
            assert isinstance(ver, (str, int))
        assert ct2 > ct1
        assert v2 > v1

    def test_bump_int_version(self, int_version):
        """ ensure bumping an integer version can happen """
        assert int_version.version == "1"

    def test_bump_version_on_bad_object(self):
        """ ensure bumping the version on a non-obspy object doesnt
         error and doesnt add creation_info """
        test_obj = "some_string"
        bump_creation_version(test_obj)
        assert not hasattr(test_obj, "creation_info")


class TestGetCatalogPath:
    """ tests for extracting save paths from catalogs """

    # a list of tuples containing names of fixtures for events/path
    bucket = [
        ("catalog_with_comment", "path_from_catalog_with_comment"),
        ("catalog_no_comment", "path_from_catalog_no_comment"),
    ]

    # fixtures
    @pytest.fixture
    def catalog_with_comment(self):
        """ return a events with a seisan comment that should dictate
        save path """
        cat = obspy.read_events()
        cat.events = [cat.events[0]]
        comment = (
            " OLDACT:NEW 12-04-04 14:21 OP:EW   STATUS:               "
            "ID:20120404142142     3"
        )
        com = Comment
        com.text = comment
        cat[0].comments.append(com)
        return cat

    @pytest.fixture
    def path_from_catalog_with_comment(self, catalog_with_comment):
        """ return the path generated by the events with comment """
        return get_event_path(catalog_with_comment)

    @pytest.fixture
    def catalog_no_comment(self):
        """ return the first event of the default events unmodified """
        cat = obspy.read_events()
        cat.events = [cat.events[0]]
        return cat

    @pytest.fixture
    def path_from_catalog_no_comment(self, catalog_no_comment):
        """ return the path from the events with no comments """
        return get_event_path(catalog_no_comment)

    @pytest.fixture(params=bucket)
    def catalog_and_path(self, request):
        """ a metafixture for gathering the events and produced paths """
        cat = request.getfixturevalue(request.param[0])
        path = request.getfixturevalue(request.param[1])
        return cat, path

    # tests
    def test_directory_structure(self, path_from_catalog_with_comment):
        """ ensure the correct directory structure was created """
        split = path_from_catalog_with_comment.split(os.sep)
        assert split[1] == "2012"
        assert split[2] == "04"

    def test_path_from_utc(self, path_from_catalog_no_comment, catalog_no_comment):
        """ ensure the correct UTC time is part of path """
        time_str = path_from_catalog_no_comment.split(os.sep)[-1]
        path_time = obspy.UTCDateTime(time_str.split(".")[0][:-6])
        cat_time = catalog_no_comment[0].origins[0].time
        assert (path_time.timestamp - cat_time.timestamp) < 1.0

    def test_resource_id_included(self, catalog_and_path):
        """ make sure the last 5 letters of the event id are in the path """
        cat, path = catalog_and_path
        last_5 = cat[0].resource_id.id[-5:]
        assert last_5 in path


class TestCatalogToDirectory:
    """ tests for converting events objects to directories """

    def test_files_created(self, tmpdir):
        """ ensure a file is created for each event in default events,
         and the bank index as well. """
        cat = obspy.read_events()
        path = Path(tmpdir)
        catalog_to_directory(cat, tmpdir)
        qml_files = list(path.rglob("*.xml"))
        assert len(qml_files) == len(cat)
        ebank = obsplus.EventBank(path)
        assert Path(ebank.index_path).exists()

    def test_events_different_time_same_id_not_duplicated(self, tmpdir):
        """ events with different times but the same id should not be
        duplicated; the old path should be used when detected. """
        cat = obspy.read_events()
        path = Path(tmpdir)
        catalog_to_directory(cat, path)
        first_event_path = get_event_path(cat[0], str(path))
        file_event_count = list(path.rglob("*.xml"))
        # modify first event preferred origin time slightly
        event = cat[0]
        origin = get_preferred(event, "origin")
        origin.time += 10
        # save to disk again
        catalog_to_directory(cat, path)
        # ensure event count didnt change
        assert len(file_event_count) == len(list(path.rglob("*.xml")))
        assert Path(first_event_path).exists()
        # read first path and make sure origin time was updated
        cat2 = obspy.read_events(str(first_event_path))
        assert len(cat2) == 1
        assert get_preferred(cat2[0], "origin").time == origin.time

    def test_from_path(self, tmpdir):
        """ catalog_to_directory should work with a path to a events. """
        cat = obspy.read_events()
        path = Path(tmpdir) / "events.xml"
        path_out1 = path.parent / "catalog_dir1"
        path_out2 = path.parent / "catalog_dir2"
        # a slightly invalid uri is used, just ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cat.write(str(path), "quakeml")
        # test works with a Path instance
        catalog_to_directory(path, path_out1)
        assert path_out1.exists()
        assert not obsplus.EventBank(path_out1).read_index().empty
        # tests with a string
        catalog_to_directory(str(path), path_out2)
        assert path_out2.exists()
        assert not obsplus.EventBank(path_out2).read_index().empty


class TestGetPreferred:
    """
    Tests for getting preferred things form events.
    """

    def test_events_no_preferred(self):
        """ Test that the last origin gets returned. """
        event = obspy.read_events()[0]
        event.preferred_origin_id = None  # clear origin_id
        assert event.origins[-1] == get_preferred(event, "origin")

    def test_preferred_no_origins(self):
        """ when the preferred id is set but origin is empty None should be
        returned. """
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


class TestMakeOrigins:
    """ Tests for the ensure origin function. """

    @pytest.fixture(scope="class")
    def inv(self):
        ds = obsplus.load_dataset("crandall")
        return ds.station_client.get_stations()

    @pytest.fixture(scope="class")
    def cat_only_picks(self, crandall_dataset):
        """ Return a catalog with only picks, no origins or magnitudes """
        cat = crandall_dataset.event_client.get_events().copy()
        for event in cat:
            event.preferred_origin_id = None
            event.preferred_magnitude_id = None
            event.origins.clear()
            event.magnitudes.clear()
        return cat

    @pytest.fixture(scope="class")
    def cat_added_origins(self, cat_only_picks, inv):
        """ run make_origins on the catalog with only picks and return """
        # get corresponding inventory
        return make_origins(events=cat_only_picks, inventory=inv)

    @pytest.fixture(scope="class")
    def strange_picks_added_origins(self, inv):
        """ make sure "rejected" picks and oddball phase hints get skipped """
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
        """ ensure all the events do indeed have origins """
        for event in cat_added_origins:
            assert event.origins, f"{event} has no origins"

    def test_origins_have_time_and_location(self, cat_added_origins):
        """ all added origins should have both times and locations. """
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
        assert ori.time == time


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
        seed = obsplus.events.utils.get_seed_id(station_mag_cont)
        assert seed == "AA.BBB.CC.DDD"

    def test_no_seed_id(self):
        """Make sure raises AttributeError if no seed info found"""
        with pytest.raises(AttributeError):
            obsplus.events.utils.get_seed_id(obspy.core.event.Pick())

    def test_unsupported(self):
        """Make sure an unsupported object raises TypeError"""
        with pytest.raises(TypeError):
            obsplus.events.utils.get_seed_id(obspy.core.event.Origin())
