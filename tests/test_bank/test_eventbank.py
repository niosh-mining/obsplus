"""
tests for event wavebank
"""
import os
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import obspy
import obspy.core.event as ev
import pandas as pd
import pytest
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees

import obsplus
import obsplus.utils.misc
from obsplus.constants import EVENT_DTYPES
from obsplus import EventBank, copy_dataset
from obsplus.utils.events import get_preferred
from obsplus.utils.testing import instrument_methods
from obsplus.utils.misc import suppress_warnings


def try_permission_sleep(callable, *args, _count=0, **kwargs):
    """A super nasty hack to get around intermittent windows permission errors"""
    try:
        return callable(*args, **kwargs)
    except PermissionError:
        time.sleep(0.01)
        if _count > 10:
            raise
    return try_permission_sleep(callable, *args, _count=_count + 1, **kwargs)


# ----------- module fixtures


def make_bank_from_catalog(path, catalog, update_index=True):
    """ make a bank from a path and a given catalog. """
    return EventBank(path).put_events(catalog, update_index=update_index)


@pytest.fixture
def cat_with_descs():
    """ Create a catalog with some simple descriptions added. """
    cat = obspy.read_events()
    descs = ["LR", "LR", "SomeSillyEvent"]
    for event, desc_txt in zip(cat, descs):
        desc = ev.EventDescription(desc_txt)
        event.event_descriptions.insert(0, desc)
    return cat


@pytest.fixture
def ebank(tmpdir, cat_with_descs):
    """ Create a bank from the default catalog. """
    cat = cat_with_descs
    path = Path(tmpdir) / "events"
    return make_bank_from_catalog(path, cat)


@pytest.fixture
def ebank_no_index(ebank):
    """ return an event bank with no index file. """
    with suppress((FileNotFoundError, PermissionError)):
        Path(ebank.index_path).unlink()
    return ebank


@pytest.fixture(scope="class")
def bing_ebank(bingham_dataset, tmpdir_factory):
    """
    Create a copy of the bingham_test data set. Then return an inited event bank
    using the temporary bingham_test bank
    """
    new = Path(str(tmpdir_factory.mktemp("bingham_test")))
    copy_dataset("bingham_test", new)
    path = new / "bingham_test" / "events"
    ebank = EventBank(path)
    ebank.update_index()
    return ebank


@pytest.fixture
def ebank_with_bad_files(tmpdir):
    """
    Create an event bank with bad files, ensure it doesn't choke indexer.
    """
    bank = EventBank(Path(tmpdir))
    cat = obspy.read_events()
    bank.put_events(cat)
    # add stream file
    st = obspy.read()
    stream_path = Path(bank.bank_path)
    st.write(str(stream_path / "not_an_event.xml"), "mseed")
    bank = EventBank(stream_path)
    # should issue warning
    with pytest.warns(UserWarning):
        bank.update_index()
    return bank


# ------------ tests


class TestBankBasics:
    """Tests for basics of the banks."""

    expected_attrs = ["update_index", "read_index"]
    low_version_str: str = "0.0.-1"

    @pytest.fixture
    def ebank_low_version(self, tmpdir, monkeypatch):
        """ return the default bank with a negative version number. """
        # monkey patch obsplus version so that a low version is saved to disk
        monkeypatch.setattr(obsplus, "__version__", self.low_version_str)
        cat = obspy.read_events()
        ebank = make_bank_from_catalog(tmpdir, cat, update_index=False)
        # write index with negative version
        with suppress_warnings():
            ebank.update_index()
        monkeypatch.undo()
        assert ebank._index_version == self.low_version_str
        assert obsplus.__version__ != self.low_version_str
        return ebank

    @pytest.fixture(scope="class")
    def ebank_with_event_no_time(self, tmp_path_factory):
        """ Create an event bank which has one file with no time. """
        tmp_path = Path(tmp_path_factory.mktemp("basic"))
        cat = obspy.read_events()
        # clear origin from first event and add an empty one
        cat[0].origins.clear()
        new_origin = ev.Origin()
        cat[0].origins.append(new_origin)
        cat[0].preferred_origin_id = new_origin.resource_id
        # now save the events
        for num, event in enumerate(cat):
            path = tmp_path / f"{num}.xml"
            event.write(path, format="quakeml")
        # init, update, return bank
        return obsplus.EventBank(tmp_path).update_index()

    def test_has_attrs(self, bing_ebank):
        """ ensure all the required attrs exist """
        for attr in self.expected_attrs:
            assert hasattr(bing_ebank, attr)

    def test_read_index(self, bing_ebank, bingham_catalog):
        """
        Read index, ensure its length matches events and id sets are
        equal.
        """
        df = bing_ebank.read_index()
        assert isinstance(df, pd.DataFrame)
        assert len(bingham_catalog) == len(df)

    def test_read_timestamp(self, bing_ebank):
        """ read the current timestamp (after index has been updated)"""
        bing_ebank.update_index()
        ts = bing_ebank.last_updated_timestamp
        assert isinstance(ts, float)
        assert ts > 0

    def test_init_bank_with_bank(self, bing_ebank):
        """ ensure an EventBank can be init'ed with an event bank. """
        ebank2 = EventBank(bing_ebank)
        assert isinstance(ebank2, EventBank)

    def test_bank_issubclass(self):
        """ ensure events is not a subclass of EventBank. """
        assert not issubclass(obspy.Catalog, EventBank)

    def test_bad_files_skipped(self, ebank_with_bad_files):
        """ make sure the index is still returnable and of the expected size. """
        cat = obspy.read_events()
        df = ebank_with_bad_files.read_index()
        assert len(cat) == len(df)

    def test_service_version(self, bing_ebank):
        """ The get_service_version method should return obsplus version """
        # first delete the old index and re-index, in case it is leftover
        # from a previous version.
        os.remove(bing_ebank.index_path)
        bing_ebank.update_index()
        assert bing_ebank.get_service_version() == obsplus.__version__

    def test_get_events_unindexed_bank(self, ebank_no_index, cat_with_descs):
        """ Calling get_waveforms on an empty bank should update index. """
        bank = ebank_no_index
        cat1_dict = {str(x.resource_id): x for x in cat_with_descs}
        # now get events and assert equal to input (although order can change)
        cat2_dict = {str(x.resource_id): x for x in bank.get_events()}
        assert cat2_dict == cat1_dict

    def test_update_index_recreates_index(self, ebank_low_version):
        """
        If the min version of the event bank is not met the index should
        be deleted and re-created. A warning should be issued.
        """
        ebank = ebank_low_version
        ipath = Path(ebank.index_path)
        mtime1 = ipath.stat().st_mtime
        with pytest.warns(UserWarning) as w:
            try_permission_sleep(ebank.update_index)
        assert len(w)  # a warning should have been raised
        mtime2 = ipath.stat().st_mtime
        # ensure the index was deleted and rewritten
        assert mtime1 < mtime2

    def test_get_events_recreates_index(self, ebank_low_version):
        """
        Not just updating the index but also initing a new bank and using it.
        """
        ebank = ebank_low_version
        # The index should not yet have been updated
        assert ebank._index_version == self.low_version_str
        with pytest.warns(UserWarning):
            ebank2 = try_permission_sleep(EventBank, ebank.bank_path)
        _ = ebank2.get_events(limit=1)
        # but after creating a new bank it should
        assert ebank._index_version == obsplus.__version__
        assert ebank2._index_version == obsplus.__version__

    def test_limit_keyword(self, ebank):
        """ Test that the limit keyword limits results (see #19) """
        limit = 2
        cat = ebank.get_events()
        assert len(cat) > limit
        cat2 = ebank.get_events(limit=limit)
        assert len(cat2) == limit
        assert cat.events[0:limit] == cat2.events

    def test_can_put_to_empty_bank(self, tmpdir):
        """
        An empty bank should be init'able created when an event is put into
        it.
        """
        path = Path(tmpdir) / "new_bank"
        bank = EventBank(path)
        assert not path.exists()
        cat = obspy.read_events()
        bank.put_events(cat)
        assert path.exists()
        assert len(bank.read_index()) == 3

    def test_index_version(self, ebank):
        """Ensure the index version returns the obsplus version."""
        assert ebank._index_version == obsplus.__version__

    def test_update_index_returns_self(self, ebank):
        """Ensure update index returns the instance for chaining."""
        out = ebank.update_index()
        assert out is ebank

    def test_events_no_time(self, ebank_with_event_no_time):
        """Tests for events which have no event time. """
        bank = ebank_with_event_no_time
        # not starttime/endtime should return all row, one has NaT
        ind = bank.read_index()
        assert len(ind) == 3
        assert ind["time"].isnull().sum() == 1
        # if any starttime/endtime are specified it should not return NaT row
        ind = bank.read_index(starttime="2012-01-01")
        assert len(ind) == 2
        assert not ind["time"].isnull().sum()
        ind = bank.read_index(endtime="2020-01-01")
        assert len(ind) == 2
        assert not ind["time"].isnull().sum()
        ind = bank.read_index(starttime="2000-01-01", endtime="2020-01-01")
        assert len(ind) == 2
        assert not ind["time"].isnull().sum()

    def test_limit_index(self, tmpdir):
        """ Test to run the logic for limiting index memory usage. """
        # This tests is a bit invasive but I cant think of a better way
        cat = obspy.read_events()
        # create temporary directory of event files
        td = Path(tmpdir)
        bank = EventBank(td).put_events(cat)
        # instrument bank, delete index, create new index
        with instrument_methods(bank) as ibank:
            os.remove(bank.index_path)
            ibank._max_events_in_memory = 1
            ibank.update_index()
            counter = ibank._counter
        # count the number of _write_update calls
        assert counter["update_index"] == 1
        assert counter["_write_update"] == 3

    def test_index_subpaths_directories(self, ebank_no_index):
        """ Ensure using subpaths only indexes certain files. """
        bank_path = Path(ebank_no_index.bank_path)
        # make new subdirectories
        for year in {"2019", "2018"}:
            (bank_path / year).mkdir(exist_ok=True)
        # put a new event in the bank
        new_events = obspy.read_events()
        new_events[0].origins[0].time = obspy.UTCDateTime("2019-01-01")
        new_events[1].origins[0].time = obspy.UTCDateTime("2018-01-01")
        new_events[0].write(str(bank_path / "2019" / "2019.xml"), "quakeml")
        new_events[1].write(str(bank_path / "2018" / "2019.xml"), "quakeml")
        # partial update index, read index
        ebank_no_index.update_index(paths=("2019", Path("2018")))
        df = ebank_no_index.read_index()
        # ensure only expected years are found
        assert len(df) == 2
        assert set(df["time"].dt.year.unique()) == {2019, 2018}

    def test_index_subpaths_files(self, ebank_no_index):
        """ Ensure a file (not just directory) can be used. """
        paths = [f"file_{x}.xml" for x in range(1, 4)]
        base_path = Path(ebank_no_index.bank_path)
        # save a single new file in bank
        for event, path in zip(obspy.read_events(), paths):
            # change origin time to avoid confusion with original files.
            event.origins[0].time = obspy.UTCDateTime("2020-01-03T19")
            event.write(str(base_path / path), "quakeml")
        # create paths mixing str, Path instance, relative, absolute
        in_paths = [base_path / paths[0], str(base_path / paths[1]), paths[2]]
        # update index, make sure everything is as expected
        df = ebank_no_index.update_index(paths=in_paths).read_index()
        assert len(df) == 3
        assert set(df["time"].dt.year) == {2020}

    def test_read_index_column_order(self, ebank):
        """The columns should be ordered according to EVENT_DTYPES."""
        df = ebank.read_index()
        overlapping_cols = set(df.columns) & set(EVENT_DTYPES)
        disjoint_cols = set(df.columns) - set(EVENT_DTYPES)
        expected_order_1 = [x for x in EVENT_DTYPES if x in overlapping_cols]
        expected_order_2 = sorted(disjoint_cols)
        expected_order = expected_order_1 + expected_order_2
        assert [str(x) for x in df.columns] == list(expected_order)

    def test_updated_event_times(self, ebank):
        """Ensure updated event times belong to this century see #146"""
        df = ebank.read_index()
        sensible_start = np.datetime64("2000-01-01T01:00:00")
        assert (df["updated"] > sensible_start).all()


class TestEventIdInBank:
    """ Tests for determining if ids are in the bank. """

    def test_event_id_in_bank(self, ebank):
        """Ensure multiple IDs can be used to query the bank."""
        cat = obspy.read_events()
        ids = [str(x.resource_id) for x in cat]
        assert set(ebank.ids_in_bank(ids)) == set(ids)

    def test_ids_not_in_bank(self, ebank):
        """Tests for ids that are not in the bank."""
        bad_ids = ["hey", "this isnt", "in the bank!"]
        assert ebank.ids_in_bank(bad_ids) is not None
        assert not ebank.ids_in_bank(bad_ids)

    def test_mixed_ids(self, ebank):
        """Tests for a mix of ids that are and arent in the bank."""
        cat = obspy.read_events()
        first_id = str(cat[0].resource_id)
        ids = [first_id, "bad_id"]
        assert set(ebank.ids_in_bank(ids)) == {first_id}


class TestReadIndexQueries:
    """ tests for index querying """

    def test_query_min_magnitude(self, bing_ebank):
        """ test min mag query """
        min_mag = 1.1
        df = bing_ebank.read_index(minmagnitude=min_mag)
        assert (df["magnitude"] >= min_mag).all()

    def test_query_max_magnitude(self, bing_ebank):
        """ test min mag query """
        max_mag = 1.1
        df = bing_ebank.read_index(maxmagnitude=max_mag)
        assert (df["magnitude"] <= max_mag).all()

    def test_query_magnitudes_with_nan(self, bing_ebank):
        """ test querying a column with NaNs (local_mag) """
        minmag, maxmag = 1.0, 1.5
        df = bing_ebank.read_index(minlocal_magnitude=minmag, maxlocal_magnitude=maxmag)
        con1 = df.local_magnitude > minmag
        con2 = df.local_magnitude < maxmag
        assert (con1 & con2).all()

    def test_query_event_id(self, bing_ebank, bingham_catalog):
        """ test query on an event id """
        eve_id = str(bingham_catalog[0].resource_id)
        df = bing_ebank.read_index(event_id=eve_id)
        assert len(df) == 1
        assert eve_id in set(df["event_id"])

    def test_query_resource_id(self, bing_ebank, bingham_catalog):
        """ test query on a resource id """
        eve_id = bingham_catalog[0].resource_id
        df = bing_ebank.read_index(event_id=eve_id)
        assert len(df) == 1
        assert str(eve_id) in set(df["event_id"])

    def test_query_event_ids(self, bing_ebank, bingham_catalog):
        """
        test querying multiple ids (specifically using something other
        than a list)
        """
        eve_ids = bing_ebank.read_index()["event_id"].iloc[0:2]
        df = bing_ebank.read_index(eventid=eve_ids)
        assert len(df) == 2
        assert df["event_id"].isin(eve_ids).all()

    def test_bad_param_raises(self, bing_ebank):
        """ assert bad query param will raise """
        with pytest.raises(ValueError):
            bing_ebank.read_index(minradius=20)

    def test_query_circular(self, bing_ebank):
        """Test circular queries in bank."""
        latitude, longitude, minradius, maxradius = (40.5, -112.12, 0.035, 0.05)
        with suppress_warnings():  # suppress install geographiclib warning
            df = bing_ebank.read_index(
                latitude=latitude,
                longitude=longitude,
                maxradius=minradius,
                minradius=maxradius,
            )
        for lat, lon in zip(df["latitude"], df["longitude"]):
            dist, _, _ = gps2dist_azimuth(latitude, longitude, lat, lon)
            assert minradius <= kilometer2degrees(dist / 1000.0) <= maxradius

    def test_query_circular_bad_params(self, bing_ebank):
        """Check that latitude, longitude can't be used with minlatitude etc"""
        with pytest.raises(ValueError):
            bing_ebank.read_index(latitude=12, minlatitude=13)

    def test_no_none_strs(self, bing_ebank):
        """
        There shouldn't be any None strings in the df.
        These should have been replaced with proper None values.
        """
        df = bing_ebank.read_index()
        assert "None" not in df.values

    def test_event_description_as_set(self, ebank):
        """
        The event description should be usable as a set, list, np array etc.
        """
        # get index with no filtering
        df_raw = ebank.read_index()
        # test filtering params
        filts = [{"LR"}, "LR", np.array(["LR"]), ["LR"]]
        for filt in filts:
            # filter with dataframe
            df_filt = [filt] if isinstance(filt, str) else filt
            df1 = df_raw[df_raw["event_description"].isin(df_filt)]
            df2 = ebank.read_index(event_description=filt)
            assert len(df1) == len(df2)
            assert df1.reset_index(drop=True).equals(df2.reset_index(drop=True))

    def test_index_time_columns(self, ebank):
        """ Ensure the time columns are pandas datetimes.  """
        df = ebank.read_index()
        # now select on datetime
        sub = df.select_dtypes([np.datetime64])
        assert {"time", "creation_time", "updated"}.issubset(sub.columns)


class TestGetEvents:
    """ tests for pulling events out of the bank """

    def test_no_params(self, bing_ebank, bingham_catalog):
        """ ensure a basic query can get an event """
        cat = bing_ebank.get_events()
        ev1 = sorted(cat.events, key=lambda x: str(x.resource_id))
        ev2 = sorted(bingham_catalog.events, key=lambda x: str(x.resource_id))
        assert ev1 == ev2

    def test_query(self, bing_ebank, bingham_catalog):
        """ test a query """
        t2 = obspy.UTCDateTime("2013-04-10")
        t1 = obspy.UTCDateTime("2010-01-01")
        cat = bing_ebank.get_events(endtime=t2, starttime=t1)
        assert cat == bingham_catalog.get_events(starttime=t1, endtime=t2)

    def test_query_circular(self, bing_ebank, bingham_catalog):
        """ The bank query should return the same as get_events on catalog. """
        latitude, longitude, minradius, maxradius = (40.5, -112.12, 0.035, 0.05)
        with suppress_warnings():
            cat1 = bing_ebank.get_events(
                latitude=latitude,
                longitude=longitude,
                maxradius=minradius,
                minradius=maxradius,
            )
            cat2 = bingham_catalog.get_events(
                latitude=latitude,
                longitude=longitude,
                maxradius=minradius,
                minradius=maxradius,
            )
        assert cat1 == cat2

    def test_issue_30(self, crandall_dataset):
        """ ensure eventid can accept a numpy array. see #30. """
        ds = crandall_dataset
        ebank = obsplus.EventBank(ds.event_path)
        # get first two event_ids
        inds = ebank.read_index()["event_id"].values[0:2]
        # query with inds as np array
        assert len(ebank.get_events(eventid=np.array(inds))) == 2


class TestPutEvents:
    """ tests for putting events into the bank """

    def test_put_new_events(self, bing_ebank):
        """ ensure a new event can be put into the bank """
        ori = ev.Origin(time=obspy.UTCDateTime("2016-01-01"))
        event = ev.Event(origins=[ori])
        event.origins[0].depth_errors = None  # see obspy 2173
        bing_ebank.put_events(event)
        event_out = bing_ebank.get_events(event_id=event.resource_id)
        assert len(event_out) == 1
        assert event_out[0] == event

    def test_update_event(self, bing_ebank, bingham_catalog):
        """ modify and event and ensure index is updated """
        index1 = bing_ebank.read_index()
        eve = bingham_catalog[0]
        rid = str(eve.resource_id)
        # modify event
        old_lat = eve.origins[0].latitude
        new_lat = old_lat + 0.15
        eve.origins[0].latitude = new_lat
        # check event back in
        bing_ebank.put_events(eve)
        # read index, ensure event_ids are unique and have correct values
        index2 = bing_ebank.read_index()
        assert not index2["event_id"].duplicated().any()
        assert len(index1) == len(index2)
        index_lat = index2.loc[index2["event_id"] == rid, "latitude"]
        assert index_lat.iloc[0] == new_lat

    def test_files_created(self, tmpdir):
        """
        Ensure a file is created for each event in default events,
        and the bank index as well.
        """
        cat = obspy.read_events()
        bank = EventBank(Path(tmpdir)).put_events(cat)
        qml_files = list(Path(bank.bank_path).rglob("*.xml"))
        assert len(qml_files) == len(cat)
        assert Path(bank.index_path).exists()

    def test_events_different_time_same_id_not_duplicated(self, tmpdir):
        """
        Events with different times but the same id should not be
        duplicated; the old path should be used when detected.
        """
        cat = obspy.read_events()
        first_id = str(cat[0].resource_id)
        bank = obsplus.EventBank(Path(tmpdir)).put_events(cat)
        df = bank.read_index().set_index("event_id")
        # modify first event preferred origin time slightly
        event = cat[0]
        origin = get_preferred(event, "origin")
        origin.time += 10
        # save to disk again
        bank.put_events(event)
        # ensure event count didnt change
        assert len(df) == len(bank.read_index())
        # read first path and make sure origin time was updated
        cat2 = bank.get_events(event_id=first_id)
        assert len(cat2) == 1
        assert get_preferred(cat2[0], "origin").time == origin.time

    def test_from_path(self, tmpdir):
        """Put events should work with a path to a directory of events."""
        cat = obspy.read_events()
        event_path = Path(tmpdir) / "events.xml"
        bank1 = EventBank(event_path.parent / "catalog_dir1")
        bank2 = EventBank(event_path.parent / "catalog_dir2")
        # a slightly invalid uri is used, just ignore
        with suppress_warnings():
            cat.write(str(event_path), "quakeml")
        # test works with a Path instance
        bank1.put_events(event_path)
        assert Path(bank1.bank_path).exists()
        assert not bank1.read_index().empty
        # tests with a string
        bank2.put_events(str(event_path))
        assert Path(bank2.bank_path).exists()
        assert not bank2.read_index().empty

    def test_put_event_no_reference_time(self, ebank):
        """Test that putting an event with no reference time raises."""
        # get an event with no reference time and no id
        event = obspy.read_events()[0]
        event.origins.clear()
        event.preferred_origin_id = None
        event.resource_id = ev.ResourceIdentifier()
        with pytest.raises(ValueError):
            ebank.put_events(event)


class TestProgressBar:
    """ Tests for the progress bar functionality of banks. """

    @pytest.fixture
    def custom_bar(self):
        """ return a custom bar implementation. """

        class Bar:
            called = False

            def __init__(self, *args, **kwargs):
                self.finished = False

            def update(self, num):
                self.__class__.called = True

            def finish(self):
                self.finished = True
                pass

        return Bar

    @pytest.fixture()
    def bar_ebank(self, tmpdir, monkeypatch):
        """ return an event bank specifically for testing ProgressBar. """
        # set the interval and min files to 1 to ensure bar gets called
        cat = obspy.read_events()
        path = Path(tmpdir)
        for event in cat:
            file_name = str(event.resource_id)[-5:] + ".xml"
            event.write(str(path / file_name), "quakeml")
        ebank = obsplus.EventBank(path)
        monkeypatch.setattr(ebank, "_bar_update_interval", 1)
        monkeypatch.setattr(ebank, "_min_files_for_bar", 1)
        # move the index to make sure there are files to update
        assert not Path(ebank.index_name).exists()
        return ebank

    @pytest.fixture()
    def bar_ebank_bad_file(self, bar_ebank):
        """ Add a waveform file to ensure ProgressBar doesnt choke. """
        path = Path(bar_ebank.bank_path)
        st = obspy.read()
        st.write(str(path / "waveform.xml"), "mseed")
        return bar_ebank

    def test_custom_bar_class(self, bar_ebank, custom_bar):
        """ Ensure a custom update bar function works. """
        bar_ebank.update_index(bar=custom_bar)
        assert custom_bar.called

    def test_custom_bar_instances(self, bar_ebank, custom_bar):
        """ Ensure passing an instance to bar will simply use instance. """
        bar = custom_bar()
        assert not bar.finished
        bar_ebank.update_index(bar=bar)
        assert bar.finished

    def test_bad_file_raises_warning(self, bar_ebank_bad_file, custom_bar):
        """ Ensure a bad files raises warning but doesn't kill progress. """
        with pytest.warns(UserWarning):
            bar_ebank_bad_file.update_index(custom_bar)

    def test_false_disables_bar(self, bar_ebank, monkeypatch):
        """ Passing False for bar argument should disable it. """
        # we ensure the default bar isnt by monkey patching the util to get it
        state = {"called": False}

        def new_get_bar(*args, **kwargs):
            state["called"] = True
            obsplus.utils.misc.get_progressbar(*args, **kwargs)

        monkeypatch.setattr(obsplus.utils.misc, "get_progressbar", new_get_bar)

        bar_ebank.update_index(bar=False)
        assert state["called"] is False

    def test_bad_value_raises(self, bar_ebank):
        """ Passing an unsupported value to bar should raise. """
        with pytest.raises(ValueError):
            bar_ebank.update_index(bar="unsupported")


class TestConcurrency:
    """ Tests for using an executor for concurrency. """

    @pytest.fixture
    def ebank_executor(self, ebank, instrumented_thread_executor, monkeypatch):
        """ Attach the instrument threadpool executor to ebank. """
        monkeypatch.setattr(ebank, "executor", instrumented_thread_executor)
        return ebank

    def test_executor_get_events(self, ebank_executor):
        """ Ensure the threadpool map function is used for reading events. """
        # get events, ensure map is used
        _ = ebank_executor.get_events()
        counter = getattr(ebank_executor.executor, "_counter", {})
        assert counter.get("map", 0) == 1

    def test_executor_index_events(self, ebank_executor):
        """ Ensure threadpool map is used for updating the index. """
        try:
            os.remove(ebank_executor.index_path)
        except FileNotFoundError:
            pass
        ebank_executor.update_index()
        counter = getattr(ebank_executor.executor, "_counter", {})
        assert counter.get("map", 0) == 1
