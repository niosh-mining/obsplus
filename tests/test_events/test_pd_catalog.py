"""
Tests for converting catalogs to dataframes.
"""
import os
import tempfile
from os.path import join, exists
from pathlib import Path

import obspy
import obspy.core.event as ev
import pandas as pd
import pytest
from obspy import UTCDateTime

from obsplus import events_to_df, picks_to_df
from obsplus.constants import EVENT_COLUMNS, PICK_COLUMNS
from obsplus.datasets.dataloader import base_path
from obsplus.utils import get_preferred, getattrs


# ---------------- helper functions


def append_func_name(list_obj):
    """ decorator to append a function name to list_obj """

    def wrap(func):
        list_obj.append(func.__name__)
        return func

    return wrap


# --------------- tests


class TestCat2Df:
    """ test the simplified dataframe conversion """

    required_columns = EVENT_COLUMNS

    # fixtures
    @pytest.fixture(scope="class")
    def df(self, test_catalog):
        """ call the catalog2df method, return result"""
        return events_to_df(test_catalog.copy())

    # tests
    def test_method_exists(self, test_catalog):
        """ test that the cat_name has the catalog2df method """
        assert hasattr(test_catalog, "to_df")

    def test_output_type(self, df):
        """ make sure a df was returned """
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, df):
        """ make sure the required columns are in the dataframe """
        assert set(df.columns).issuperset(self.required_columns)

    def test_lengths(self, df, test_catalog):
        """ ensure the lengths of the dataframe and events are the same """
        assert len(df) == len(test_catalog)

    def test_event_description(self, df, test_catalog):
        """ ensure the event descriptions match """
        for eve, description in zip(test_catalog, df.event_description):
            if eve.event_descriptions:
                ed = eve.event_descriptions[0].text
            else:
                ed = None
            assert ed == description

    def test_str(self):
        """ ensure there is a string rep for catalog_to_df. """
        cat_to_df_str = str(events_to_df)
        assert isinstance(cat_to_df_str, str)  # dumb test to boost coverage

    def test_event_id_in_columns(self, df):
        """ Sometime the event_id was changed to the index, make sure it is
        still a column. """
        cols = df.columns
        assert "event_id" in cols


class TestCat2DfPreferreds:
    """ Make sure the preferred origins/mags show up in the df """

    # fixtures
    @pytest.fixture(scope="class", autouse=True)
    def preferred_magnitudes(self, test_catalog):
        """ set the preferred magnitudes to the first magnitudes,
        return list of magnitudes """
        mags = []
        for eve in test_catalog:
            if len(eve.magnitudes):
                magid = eve.magnitudes[0].resource_id.id
                mags.append(eve.magnitudes[0])
            else:
                magid = None
                mags.append(None)
            eve.preferred_magnitude_id = magid
        return mags

    @pytest.fixture(scope="class", autouse=True)
    def preferred_origins(self, test_catalog):
        """ set the preferred magnitudes to the first magnitudes,
        return list of magnitudes """
        origins = []
        for eve in test_catalog:
            if len(eve.origins):
                orid = eve.origins[0].resource_id
                origins.append(eve.origins[0])
            else:
                orid = None
                origins.append(None)
            eve.preferred_origin_id = orid
        return origins

    @pytest.fixture(scope="class")
    def df(self, test_catalog):
        """ call the catalog2df method, return result"""
        out = events_to_df(test_catalog.copy())
        out.reset_index(inplace=True, drop=True)
        return out

    # tests
    def test_origins(self, df, preferred_origins):
        """ ensure the origins are correct """
        for ind, row in df.iterrows():
            origin = preferred_origins[ind]
            assert origin.latitude == row.latitude
            assert origin.longitude == row.longitude
            assert origin.time == obspy.UTCDateTime(row.time)

    def test_magnitudes(self, df, preferred_magnitudes):
        """ ensure the origins are correct """
        for ind, row in df.iterrows():
            mag = preferred_magnitudes[ind]
            assert mag.mag == row.magnitude
            mtype1 = str(row.magnitude_type).upper()
            mtype2 = str(mag.magnitude_type).upper()
            assert mtype1 == mtype2


class TestReadEvents:
    """ ensure events can be read in """

    fixtures = []

    # fixtures
    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def events_from_catalog(self):
        """ read events from a events object """
        cat = obspy.read_events()
        return events_to_df(cat)

    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def events_from_dataframe(self):
        event_dict = {
            "time": obspy.UTCDateTime(),
            "latitude": 41,
            "longitude": -111.1,
            "depth": 10.0,
            "magnitude": 4.5,
        }
        df = pd.DataFrame(pd.Series(event_dict)).T
        return events_to_df(df)

    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def event_df_subset(self, kem_archive):
        """ read in the partial list of events """
        path = join(Path(kem_archive).parent, "catalog_subset.csv")
        return events_to_df(path)

    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def rewritten_file_event_df(self, event_df_subset):
        """ write the event_df to disk and try to read it in again """
        with tempfile.NamedTemporaryFile() as tf:
            event_df_subset.to_csv(tf.name)
            yield events_to_df(tf.name)
        if os.path.exists(tf.name):  # clean up temp file if needed
            os.remove(tf.name)

    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def catalog_no_magnitude(self):
        """ get a events with no magnitudes (should just fill with NaN) """
        t1 = obspy.UTCDateTime("2099-04-01T00-01-00")
        ori = ev.Origin(time=t1, latitude=47.1, longitude=-100.22)
        event = ev.Event(origins=[ori])
        cat = ev.Catalog(events=[event])
        return events_to_df(cat)

    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def catalog_empty(self):
        """ get a with one blank event """
        event = ev.Event()
        cat = ev.Catalog(events=[event])
        return events_to_df(cat)

    @pytest.fixture(scope="class", params=fixtures)
    def read_events_output(self, request):
        """ the parametrized output of read_events fixtures """
        return request.getfixturevalue(request.param)

    @pytest.fixture
    def events_rejected_picks(self, bingham_dataset):
        """ return a events that has all rejected picks """
        cat = bingham_dataset.event_client.get_events().copy()
        for ev in cat:
            for pick in ev.picks:
                pick.evaluation_status = "rejected"
        return cat

    # tests
    def test_basics(self, read_events_output):
        """ make sure a dataframe is returned """
        assert isinstance(read_events_output, pd.DataFrame)
        assert len(read_events_output)

    def test_rejected_phases_still_counted(self, events_rejected_picks):
        """ ensure rejected picks are still counted in arrival numbering """
        df = events_to_df(events_rejected_picks)
        assert (df.p_phase_count != 0).all()


class TestReadKemEvents:
    """ test for reading a variety of pick formats from the KEM_TESTCASE dataset """

    dataset_params = ["events.xml", "catalog.csv"]

    # fixtures
    @pytest.fixture(scope="class", params=dataset_params)
    def cat_df(self, request, kem_archive):
        """ collect all the supported inputs are parametrize"""
        return events_to_df(base_path / "kemmerer" / request.param)

    @pytest.fixture(scope="class")
    def catalog(self, kem_archive):
        """ return the events """
        return obspy.read_events(str(base_path / "kemmerer" / "events.xml"))

    # tests
    def test_len(self, cat_df, catalog):
        """ ensure the correct number of items was returned """
        assert len(cat_df) == len(catalog.events)

    def test_column_order(self, cat_df):
        """ ensure the order of the columns is correct """
        cols = list(cat_df.columns)
        assert list(EVENT_COLUMNS) == cols[: len(EVENT_COLUMNS)]

    def test_cat_to_df_method(self):
        """ ensure the events object has the to_df method bolted on """
        cat = obspy.read_events()
        df = cat.to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(cat)


class TestReadPhasePicks:
    """ ensure phase picks can be read in """

    fixtures = []
    cat_name = "2016-04-13T23-51-13.xml"

    @pytest.fixture(scope="class")
    def tcat(self, catalog_cache):
        """ retad in a events for testing """
        return catalog_cache[self.cat_name]

    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def catalog_output(self, tcat):
        """ call read_picks on the events, return result """
        return picks_to_df(tcat)

    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def dataframe_output(self, tcat):
        """ return read_picks result from reading dataframe """
        df = picks_to_df(tcat)
        return picks_to_df(df)

    @pytest.fixture(scope="class")
    def empty_catalog(self):
        """ run an empty events through the picks_to_df function """
        event = ev.Event()
        cat = obspy.Catalog(events=[event])
        return picks_to_df(cat)

    @pytest.fixture(scope="class")
    def picks_no_origin(self):
        """ create a events that has picks but no origin """
        t0 = UTCDateTime("2016-01-01T10:12:15.222")

        def wave_id(seed_str):
            return ev.WaveformStreamID(seed_string=seed_str)

        picks = [
            ev.Pick(time=t0 + 2, waveform_id=wave_id("UU.TMU..HHZ")),
            ev.Pick(time=t0 + 1.2, waveform_id=wave_id("UU.BOB.01.ELZ")),
            ev.Pick(time=t0 + 3.2, waveform_id=wave_id("UU.TEX..EHZ")),
        ]
        return picks_to_df(ev.Event(picks=picks))

    @pytest.fixture(scope="class", params=fixtures)
    def read_picks_output(self, request):
        """ return the outputs from the fixtures """
        return request.getfixturevalue(request.param)

    @pytest.fixture
    def bingham_cat_only_picks(self, bingham_dataset):
        """ return bingham catalog with everything but picks removed """
        events = []
        for eve in bingham_dataset.event_client.get_events().copy():
            events.append(ev.Event(picks=eve.picks))
        return obspy.Catalog(events=events)

    # general tests
    def test_type(self, read_picks_output):
        """ make sure a dataframe was returned """
        assert isinstance(read_picks_output, pd.DataFrame)

    def test_len(self, read_picks_output, tcat):
        """ req_len should be the same as the picks req_len in events """
        assert len(tcat[0].picks) == len(read_picks_output)

    # empty_catalog_tests
    def test_empty_catalog_input(self, empty_catalog):
        """ ensure a zero len dataframe was returned with required columns """
        assert isinstance(empty_catalog, pd.DataFrame)
        assert not len(empty_catalog)
        assert set(empty_catalog.columns).issubset(PICK_COLUMNS)

    def test_picks_no_origin(self, picks_no_origin):
        """ ensure not having an origin time returns min of picks per event. """
        df = picks_no_origin
        assert (df.event_time == df.time.min()).all()

    def test_unique_event_time_no_origin(self, bingham_cat_only_picks):
        """ Ensure events with no origin don't all return the same time. """
        df = picks_to_df(bingham_cat_only_picks)
        assert len(df.event_time.unique()) == len(df.event_id.unique())


class TestReadKemPicks:
    """ test for reading a variety of pick formats from the kemmerer
    dataset """

    path = base_path / "kemmerer"
    csv_path = path / "picks.csv"
    qml_path = str(path / "events.xml")
    qml = obspy.read_events(str(qml_path))
    picks = [pick for eve in qml for pick in eve.picks]
    supported_inputs = [qml_path, qml, csv_path]

    # fixtures
    @pytest.fixture(scope="class", params=supported_inputs)
    def pick_df(self, request):
        """ collect all the supported inputs are parametrize"""
        return picks_to_df(request.param)

    # tests
    def test_len(self, pick_df):
        """ ensure the correct number of items was returned """
        assert len(pick_df) == len(self.picks)

    def test_column_order(self, pick_df):
        """ ensure the order of the columns is correct """
        cols = list(pick_df.columns)
        assert list(PICK_COLUMNS) == cols[: len(PICK_COLUMNS)]

    def test_event_id(self, pick_df):
        """ ensure nan values are not in dataframe event_id column """
        assert not pick_df.event_id.isnull().any()

    def test_seed_id(self, pick_df):
        """ ensure valid seed_ids were created. """
        # recreate seed_id and make sure columns are equal
        df = pick_df
        seed = (
            df["network"]
            + "."
            + df["station"]
            + "."
            + df["location"]
            + "."
            + df["channel"]
        )
        assert (seed == df["seed_id"]).all()


class TestGetPreferred:
    def test_bad_preferred_origin(self):
        """ ensure the bad preferred just returns last in list """
        eve = obspy.read_events()[0]
        eve.preferred_origin_id = "bob"
        with pytest.warns(UserWarning) as w:
            preferred_origin = get_preferred(eve, "origin")
        assert len(w) == 1
        assert preferred_origin is eve.origins[-1]


class TestParseOrDefault:
    # tests
    def test_none_returns_empty(self):
        """ make sure None returns empty dict"""
        out = getattrs(None, ["bob"])
        assert isinstance(out, dict)
        assert not out


class TestReadDirectoryOfCatalogs:
    """ tests that a directory of quakeml files can be read """

    nest_name = "nest"

    # helper functions
    def nest_directly(self, nested_times, path):
        """ make a directory nested n times """
        nd_name = join(path, self.nest_name)
        if not exists(nd_name) and nested_times:
            os.makedirs(nd_name)
        elif not nested_times:  # recursion limit reached
            return path
        return self.nest_directly(nested_times - 1, nd_name)

    # fixtures
    @pytest.fixture(scope="class")
    def catalog_directory(self):
        """ return a directory of catalogs """
        cat = obspy.read_events()
        with tempfile.TemporaryDirectory() as tempdir:
            for num, eve in enumerate(cat.events):
                new_cat = obspy.Catalog(events=[eve])
                file_name = f"{num}.xml"
                write_path = join(self.nest_directly(num, tempdir), file_name)
                new_cat.write(write_path, "quakeml")
            yield tempdir

    @pytest.fixture(scope="class")
    def read_catalog(self, catalog_directory):
        """ return the results of calling catalog_to_df on directory """
        return events_to_df(catalog_directory)

    # tests
    def test_df_are_same(self, read_catalog):
        df = events_to_df(obspy.read_events())
        assert (df.columns == read_catalog.columns).all()
        assert len(df) == len(read_catalog)
        assert set(df.time) == set(read_catalog.time)
