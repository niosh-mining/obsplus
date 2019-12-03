"""
tests for event wavebank
"""
import os
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import obspy
import obspy.core.event as ev
import pandas as pd
import pytest
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees

import obsplus
from obsplus import EventBank, copy_dataset
from obsplus.events.utils import catalog_to_directory
from obsplus.testing import instrument_methods


# ----------- module fixtures

#
# @pytest.fixture(scope="class")
# def catalog(bingham_dataset):
#     """ return the bingham_test_case events """
#     cli = bingham_dataset.event_client
#     cat = cli.get_events()
#     assert len(cat)
#     return cat


@pytest.fixture
def ebank(tmpdir):
    """ Create a bank from the default catalog. """
    path = Path(tmpdir) / "events"
    # get catalog, add event descriptions to it
    cat = obspy.read_events()
    descs = ["LR", "LR", "SomeSillyEvent"]
    for event, desc_txt in zip(cat, descs):
        desc = ev.EventDescription(desc_txt)
        event.event_descriptions.insert(0, desc)
    obsplus.events.utils.catalog_to_directory(cat, path)
    ebank = EventBank(path)
    ebank.update_index()
    return ebank


@pytest.fixture(scope="class")
def bing_ebank(bingham_dataset, tmpdir_factory):
    """
    Create a copy of the bingham data set. Then return an inited event bank
    using the temporary bingham bank
    """
    new = Path(str(tmpdir_factory.mktemp("bingham")))
    copy_dataset("bingham", new)
    path = new / "bingham" / "events"
    ebank = EventBank(path)
    ebank.update_index()
    return ebank


@pytest.fixture
def ebank_with_bad_files(tmpdir):
    """ create an event bank with bad files, ensure it doesn't choke
    indexer. """
    path = Path(tmpdir)
    cat = obspy.read_events()
    catalog_to_directory(cat, path)
    # add stream file
    st = obspy.read()
    st.write(str(path / "not_an_event.xml"), "mseed")
    bank = EventBank(path)
    # should issue warning
    with pytest.warns(UserWarning):
        bank.update_index()
    return bank


# ------------ tests


class TestBankBasics:
    expected_attrs = ["update_index", "read_index"]
    low_version_str: str = "0.0.-1"

    @pytest.fixture
    def ebank_low_version(self, ebank, monkeypatch):
        """ return the default bank with a negative version number. """
        # monkey patch obsplus version so that a low version is saved to disk
        monkeypatch.setattr(obsplus, "__version__", self.low_version_str)
        # write index with negative version
        os.remove(ebank.index_path)
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
        """ read index, ensure its length matches events and id sets are
        equal """
        df = bing_ebank.read_index()
        assert isinstance(df, pd.DataFrame)
        assert len(bingham_catalog) == len(df)

    def test_read_timestamp(self, bing_ebank):
        """ read the current timestamp (after index has been updated)"""
        bing_ebank.update_index()
        ts = bing_ebank.last_updated
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

    def test_get_events_empty_bank(self, tmp_path):
        """ Calling get_waveforms on an empty bank should update index. """
        cat1 = obspy.read_events()
        catalog_to_directory(cat1, tmp_path, event_bank_index=False)
        cat1_dict = {str(x.resource_id): x for x in cat1}
        # get a bank, ensure it has no index and call get events
        bank = obsplus.EventBank(tmp_path)
        index = Path(bank.index_path)
        if index.exists():
            index.unlink()
        # now get events and assert equal to input (although order can change)
        cat2 = bank.get_events()
        cat2_dict = {str(x.resource_id): x for x in cat2}
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
            ebank.update_index()
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
            ebank2 = EventBank(ebank.bank_path)
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
        """ An empty bank should be init'able created when an event is
        put into it """
        path = Path(tmpdir) / "new_bank"
        bank = EventBank(path)
        assert not path.exists()
        cat = obspy.read_events()
        bank.put_events(cat)
        assert path.exists()
        assert len(bank.read_index()) == 3

    def test_index_version(self, ebank):
        """ ensure the index version returns the obsplus version. """
        assert ebank._index_version == obsplus.__version__

    def test_update_index_returns_self(self, ebank):
        """ ensure update index returns the instance for chaining. """
        out = ebank.update_index()
        assert out is ebank

    def test_events_no_time(self, ebank_with_event_no_time):
        """ Tests for events which have no event time. """
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
        kwargs = dict(cat=cat, path=td, event_bank_index=False, check_duplicates=False)
        obsplus.events.utils.catalog_to_directory(**kwargs)
        # init bank and add index
        bank = EventBank(td).update_index()
        # instrument bank, delete index, create new index
        with instrument_methods(bank) as ibank:
            os.remove(bank.index_path)
            ibank._max_events_in_memory = 1
            ibank.update_index()
            counter = ibank._counter
        # count the number of _write_update calls
        assert counter["update_index"] == 1
        assert counter["_write_update"] == 3


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
        latitude, longitude, minradius, maxradius = (40.5, -112.12, 0.035, 0.05)
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
        latitude, longitude, minradius, maxradius = (40.5, -112.12, 0.035, 0.05)
        cat = bing_ebank.get_events(
            latitude=latitude,
            longitude=longitude,
            maxradius=minradius,
            minradius=maxradius,
        )
        assert cat == bingham_catalog.get_events(
            latitude=latitude,
            longitude=longitude,
            maxradius=minradius,
            minradius=maxradius,
        )

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
    def bar_ebank(self, ebank, monkeypatch):
        """ return an event bank specifically for testing ProgressBar. """
        # set the interval and min files to 1 to ensure bar gets called
        monkeypatch.setattr(ebank, "_bar_update_interval", 1)
        monkeypatch.setattr(ebank, "_min_files_for_bar", 1)
        # move the index to make sure there are files to update
        index_path = Path(ebank.index_path)
        if index_path.exists():
            os.remove(index_path)
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
            obsplus.utils.get_progressbar(*args, **kwargs)

        monkeypatch.setattr(obsplus.utils, "get_progressbar", new_get_bar)

        bar_ebank.update_index(bar=False)
        assert state["called"] is False

    def test_bad_value_raises(self, bar_ebank):
        """ Passing an unsupported value to bar should raise. """
        with pytest.raises(ValueError):
            bar_ebank.update_index(bar="unsupported")

    def test_update_bar_default(self, bar_ebank, monkeypatch):
        """ Ensure the default progress bar shows up. """
        stringio = StringIO()
        monkeypatch.setattr(sys, "stdout", stringio)
        bar_ebank.update_index()
        stringio.seek(0)
        out = stringio.read()
        assert "updating or creating" in out


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
