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

import obsplus
from obsplus import (
    events_to_df,
    picks_to_df,
    arrivals_to_df,
    amplitudes_to_df,
    station_magnitudes_to_df,
    magnitudes_to_df,
    get_preferred,
)
from obsplus.constants import (
    EVENT_COLUMNS,
    PICK_COLUMNS,
    ARRIVAL_COLUMNS,
    AMPLITUDE_COLUMNS,
    STATION_MAGNITUDE_COLUMNS,
    MAGNITUDE_COLUMNS,
)

from obsplus.utils import getattrs, get_nslc_series

common_extractor_cols = {
    "agency_id",
    "author",
    "channel",
    "creation_time",
    "location",
    "network",
    "seed_id",
    "station",
    "event_id",
    "event_time",
}
common_obj_attrs = {"creation_info", "comments", "waveform_id"}


# ---------------- helper functions


def append_func_name(list_obj):
    """ decorator to append a function name to list_obj """

    def wrap(func):
        list_obj.append(func.__name__)
        return func

    return wrap


def pick_generator(scnls):
    picks = []
    for scnl in scnls:
        p = ev.Pick(
            time=UTCDateTime(), waveform_id=ev.WaveformStreamID(seed_string=scnl)
        )
        picks.append(p)
    return picks


def arr_generator(picks):
    counter = 1
    params = {"phase": "P"}
    arrivals = []
    picks = picks or []
    for pick in picks:
        a = ev.Arrival(
            pick_id=pick.resource_id,
            time_correction=counter * 0.05,
            azimuth=counter * 5,
            distance=counter * 0.1,
            takeoff_angle=counter * 2,
            time_residual=counter * 0.15,
            horizontal_slowness_residual=counter * 0.2,
            backazimuth_residual=counter * 0.25,
            time_weight=counter * 0.3,
            horizontal_slowness_weight=counter * 0.4,
            backazimuth_weight=counter * 0.5,
            earth_model_id=ev.ResourceIdentifier(),
            creation_info=ev.CreationInfo(
                agency_id="dummy_agency", author="dummy", creation_time=UTCDateTime()
            ),
            **params,
        )
        arrivals.append(a)
        counter += 1
    return arrivals


def amp_generator(scnls=None, picks=None):
    counter = 1
    amps = []
    scnls = scnls or []
    params = {
        "type": "A",
        "unit": "dimensionless",
        "method_id": "mag_calculator",
        "filter_id": ev.ResourceIdentifier("Wood-Anderson"),
        "magnitude_hint": "M",
        "category": "point",
        "evaluation_mode": "manual",
        "evaluation_status": "confirmed",
    }
    for scnl in scnls:
        a = ev.Amplitude(
            generic_amplitude=counter,
            generic_amplitude_errors=ev.QuantityError(
                uncertainty=counter * 0.1, confidence_level=95
            ),
            period=counter * 2,
            snr=counter * 5,
            time_window=ev.TimeWindow(0, 0.1, UTCDateTime()),
            waveform_id=ev.WaveformStreamID(seed_string=scnl),
            scaling_time=UTCDateTime(),
            scaling_time_errors=ev.QuantityError(
                uncertainty=counter * 0.001, confidence_level=95
            ),
            creation_info=ev.CreationInfo(
                agency_id="dummy_agency", author="dummy", creation_time=UTCDateTime()
            ),
            **params,
        )
        amps.append(a)
        counter += 1
    picks = picks or []
    for pick in picks:
        a = ev.Amplitude(
            generic_amplitude=counter,
            generic_amplitude_errors=ev.QuantityError(
                uncertainty=counter * 0.1, confidence_level=95
            ),
            period=counter * 2,
            snr=counter * 5,
            time_window=ev.TimeWindow(0, 0.1, UTCDateTime()),
            pick_id=pick.resource_id,
            scaling_time=UTCDateTime(),
            scaling_time_errors=ev.QuantityError(
                uncertainty=counter * 0.001, confidence_level=95
            ),
            creation_info=ev.CreationInfo(
                agency_id="dummy_agency", author="dummy", creation_time=UTCDateTime()
            ),
            **params,
        )
        amps.append(a)
        counter += 1
    return amps


def sm_generator(scnls=None, amplitudes=None):
    counter = 1
    sms = []
    scnls = scnls or []
    params = {
        "origin_id": ev.ResourceIdentifier(),
        "station_magnitude_type": "M",
        "method_id": "mag_calculator",
    }

    for scnl in scnls:
        sm = ev.StationMagnitude(
            mag=counter,
            mag_errors=ev.QuantityError(uncertainty=counter * 0.1, confidence_level=95),
            waveform_id=ev.WaveformStreamID(seed_string=scnl),
            creation_info=ev.CreationInfo(
                agency_id="dummy_agency", author="dummy", creation_time=UTCDateTime()
            ),
            **params,
        )
        sms.append(sm)
        counter += 1
    amplitudes = amplitudes or []
    for amp in amplitudes:
        sm = ev.StationMagnitude(
            mag=counter,
            mag_errors=ev.QuantityError(uncertainty=counter * 0.1, confidence_level=95),
            amplitude_id=amp.resource_id,
            creation_info=ev.CreationInfo(
                agency_id="dummy_agency", author="dummy", creation_time=UTCDateTime()
            ),
            **params,
        )
        sms.append(sm)
        counter += 1
    return sms


def mag_generator(mag_types):
    params = {
        "origin_id": ev.ResourceIdentifier(),
        "method_id": ev.ResourceIdentifier("mag_calculator"),
        "station_count": 2,
        "azimuthal_gap": 30,
        "evaluation_mode": "manual",
        "evaluation_status": "reviewed",
    }
    mags = []
    counter = 1
    for mt in mag_types:
        m = ev.Magnitude(
            mag=counter,
            magnitude_type=mt,
            mag_errors=ev.QuantityError(uncertainty=counter * 0.1, confidence_level=95),
            creation_info=ev.CreationInfo(
                agency_id="dummy_agency", author="dummy", creation_time=UTCDateTime()
            ),
            **params,
        )
        mags.append(m)
    return mags


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
        """ call the catalog2df method, return result """
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
    base_path = obsplus.load_dataset("kemmerer").source_path.parent

    # fixtures
    @pytest.fixture(scope="class", params=dataset_params)
    def cat_df(self, request, kem_archive):
        """ collect all the supported inputs are parametrize"""
        return events_to_df(self.base_path / "kemmerer" / request.param)

    @pytest.fixture(scope="class")
    def catalog(self, kem_archive):
        """ return the events """
        return obspy.read_events(str(self.base_path / "kemmerer" / "events.xml"))

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
        """ read in an event for testing """
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

    def test_read_uncertainty(self):
        """
        tests that uncertainties in time_errors attribute are read. See #55.
        """
        kwargs = dict(lower_uncertainty=1, upper_uncertainty=2, uncertainty=12)
        time_error = ev.QuantityError(**kwargs)
        waveform_id = ev.WaveformStreamID(station_code="A")
        pick = ev.Pick(
            time=UTCDateTime(), time_errors=time_error, waveform_id=waveform_id
        )
        df = picks_to_df(pick)
        assert set(kwargs).issubset(df.columns)
        assert len(df) == 1
        ser = df.iloc[0]
        assert all([ser[i] == kwargs[i] for i in kwargs])

    def test_none_onset(self):
        """
        Make sure Nones in the data get handled properly
        """
        waveform_id = ev.WaveformStreamID(station_code="A")
        pick = ev.Pick(time=UTCDateTime(), waveform_id=waveform_id)
        df = picks_to_df(pick)
        assert df.onset.iloc[0] == ""
        assert df.polarity.iloc[0] == ""


class TestReadKemPicks:
    """ test for reading a variety of pick formats from the kemmerer
    dataset """

    path = obsplus.load_dataset("kemmerer").source_path
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
        seed = get_nslc_series(pick_df)
        assert (seed == df["seed_id"]).all()


class TestReadArrivals:
    # fixtures
    @pytest.fixture(scope="class")
    def dummy_cat(self):
        scnls1 = ["UK.STA1..HHZ", "UK.STA2..HHZ"]
        cat = ev.Catalog()
        eve1 = ev.Event()
        eve1.origins.append(ev.Origin(time=UTCDateTime()))
        eve1.preferred_origin_id = eve1.origins[0].resource_id
        picks = pick_generator(scnls1)
        eve1.picks = picks
        eve1.preferred_origin().arrivals = arr_generator(picks)
        scnls2 = ["UK.STA3..HHZ", "UK.STA4..HHZ", "UK.STA5..HHZ"]
        eve2 = ev.Event()
        eve2.origins.append(ev.Origin(time=UTCDateTime()))
        eve2.preferred_origin_id = eve2.origins[0].resource_id
        picks = pick_generator(scnls2)
        eve2.picks = picks
        eve2.preferred_origin().arrivals = arr_generator(picks)
        cat.events = [eve1, eve2]
        return cat

    @pytest.fixture(scope="class")
    def empty_cat(self):
        return ev.Catalog()

    @pytest.fixture(scope="class")
    def no_origin(self):
        cat = ev.Catalog()
        cat.append(ev.Event())
        return cat

    @pytest.fixture(scope="class")
    def read_arr_output(self, dummy_cat):
        return arrivals_to_df(dummy_cat)

    @pytest.fixture(scope="class")
    def ser_dict(self, read_arr_output):
        """ values to compare from the extractor """
        ser_dict = dict(read_arr_output.iloc[0])
        for key in common_extractor_cols:
            ser_dict.pop(key, None)
        return ser_dict

    @pytest.fixture(scope="class")
    def arr_dict(self, dummy_cat):
        """ values to compare from the arrivals """
        origin = dummy_cat[0].preferred_origin()
        arr = origin.arrivals[0]
        arr_dict = dict(arr.__dict__)
        # Remove unnecessary items
        for key in common_obj_attrs:
            arr_dict.pop(key, None)
        arr_dict.pop("takeoff_angle_errors")
        # modify/add more complex items
        arr_dict["resource_id"] = arr_dict["resource_id"].id
        arr_dict["pick_id"] = arr_dict["pick_id"].id
        arr_dict["origin_id"] = origin.resource_id.id
        arr_dict["origin_time"] = origin.time.timestamp
        return arr_dict

    # general tests
    def test_type(self, read_arr_output):
        """ make sure a dataframe was returned """
        assert isinstance(read_arr_output, pd.DataFrame)

    def test_len(self, read_arr_output, dummy_cat):
        """ req_len should be the same as the sms req_len in events """
        req_len = len(dummy_cat[0].preferred_origin().arrivals) + len(
            dummy_cat[1].preferred_origin().arrivals
        )
        assert req_len == len(read_arr_output)

    def test_values(self, ser_dict, arr_dict):
        """ make sure the values of the first arrival are as expected """
        assert ser_dict == arr_dict

    # empty catalog tests
    def test_empty_catalog(self, empty_cat):
        """ ensure returns empty df with required columns """
        df = arrivals_to_df(empty_cat)
        assert isinstance(df, pd.DataFrame)
        assert not len(df)
        assert set(df.columns).issubset(ARRIVAL_COLUMNS)

    def test_no_origin(self, no_origin):
        """ ensure returns empty df with required columns """
        df = arrivals_to_df(no_origin)
        assert isinstance(df, pd.DataFrame)
        assert not len(df)
        assert set(df.columns).issubset(ARRIVAL_COLUMNS)


class TestReadAmplitudes:
    # fixtures
    @pytest.fixture(scope="class")
    def dummy_cat(self):
        scnls1 = ["UK.STA1..HHZ", "UK.STA2..HHZ"]
        cat = ev.Catalog()
        eve1 = ev.Event()
        eve1.origins.append(ev.Origin(time=UTCDateTime()))
        eve1.picks = pick_generator(scnls1)
        eve1.amplitudes = amp_generator(picks=eve1.picks)
        scnls2 = ["UK.STA3..HHZ", "UK.STA4..HHZ", "UK.STA5..HHZ"]
        eve2 = ev.Event()
        eve2.origins.append(ev.Origin(time=UTCDateTime()))
        eve2.amplitudes = amp_generator(scnls=scnls2)
        cat.events = [eve1, eve2]
        return cat

    @pytest.fixture(scope="class")
    def empty_cat(self):
        return ev.Catalog()

    @pytest.fixture(scope="class")
    def read_amps_output(self, dummy_cat):
        return amplitudes_to_df(dummy_cat)

    @pytest.fixture(scope="class")
    def amplitude(self, dummy_cat):
        return dummy_cat[0].amplitudes[0]

    @pytest.fixture(scope="class")
    def amp_series(self, read_amps_output):
        return read_amps_output.iloc[0]

    @pytest.fixture(scope="class")
    def ser_dict(self, amp_series):
        """ values to compare from the extractor """
        ser_dict = dict(amp_series)
        err_cols = {
            "confidence_level",
            "uncertainty",
            "lower_uncertainty",
            "upper_uncertainty",
        }
        for key in common_extractor_cols.union(err_cols):
            ser_dict.pop(key, None)
        return ser_dict

    @pytest.fixture(scope="class")
    def amp_dict(self, amplitude):
        """ values to compare from the arrivals """
        amp_dict = dict(amplitude.__dict__)
        # Remove unnecessary items
        err_objs = {"generic_amplitude_errors", "scaling_time_errors", "period_errors"}
        for key in common_obj_attrs.union(err_objs):
            amp_dict.pop(key, None)
        # modify/add more complex items
        amp_dict["resource_id"] = amp_dict["resource_id"].id
        amp_dict["pick_id"] = amp_dict["pick_id"].id
        amp_dict["filter_id"] = amp_dict["filter_id"].id
        amp_dict["method_id"] = amp_dict["method_id"].id
        amp_dict["scaling_time"] = amp_dict["scaling_time"].timestamp
        time_window = amp_dict.pop("time_window")
        amp_dict["reference"] = time_window.reference.timestamp
        amp_dict["time_begin"] = time_window.begin
        amp_dict["time_end"] = time_window.end
        return amp_dict

    # general tests
    def test_type(self, read_amps_output):
        """ make sure a dataframe was returned """
        assert isinstance(read_amps_output, pd.DataFrame)

    def test_len(self, read_amps_output, dummy_cat):
        """ req_len should be the same as the amps req_len in events """
        req_len = len(dummy_cat[0].amplitudes) + len(dummy_cat[1].amplitudes)
        assert req_len == len(read_amps_output)

    def test_values(self, ser_dict, amp_dict):
        """ make sure the values of the first amplitude are as expected """
        assert ser_dict == amp_dict

    def test_creation_time(self, amplitude, amp_series):
        assert amp_series["creation_time"] == obspy.UTCDateTime(
            amplitude.creation_info.creation_time
        )
        assert amp_series["author"] == amplitude.creation_info.author
        assert amp_series["agency_id"] == amplitude.creation_info.agency_id

    # empty catalog tests
    def test_empty_catalog(self, empty_cat):
        """ ensure returns empty df with required columns """
        df = amplitudes_to_df(empty_cat)
        assert isinstance(df, pd.DataFrame)
        assert not len(df)
        assert set(df.columns).issubset(AMPLITUDE_COLUMNS)


class TestReadStationMagnitudes:
    # fixtures
    @pytest.fixture(scope="class")
    def dummy_cat(self):
        scnls1 = ["UK.STA1..HHZ", "UK.STA2..HHZ"]
        cat = ev.Catalog()
        eve1 = ev.Event()
        eve1.origins.append(ev.Origin(time=UTCDateTime()))
        eve1.amplitudes = amp_generator(scnls1)
        eve1.station_magnitudes = sm_generator(amplitudes=eve1.amplitudes)
        scnls2 = ["UK.STA3..HHZ", "UK.STA4..HHZ", "UK.STA5..HHZ"]
        eve2 = ev.Event()
        eve2.origins.append(ev.Origin(time=UTCDateTime()))
        eve2.station_magnitudes = sm_generator(scnls=scnls2)
        cat.events = [eve1, eve2]
        return cat

    @pytest.fixture(scope="class")
    def dummy_mag(self):
        scnls = ["UK.STA1..HHZ", "UK.STA2..HHZ"]
        eve = ev.Event()
        sms = sm_generator(scnls=scnls)
        smcs = []
        for sm in sms:
            smcs.append(
                ev.StationMagnitudeContribution(station_magnitude_id=sm.resource_id)
            )
        mag = ev.Magnitude(mag=1, station_magnitude_contributions=smcs)
        eve.magnitudes = [mag]
        eve.station_magnitudes = sms
        return eve

    @pytest.fixture(scope="class")
    def empty_cat(self):
        return ev.Catalog()

    @pytest.fixture(scope="class")
    def read_sms_output(self, dummy_cat):
        return station_magnitudes_to_df(dummy_cat)

    @pytest.fixture(scope="class")
    def ser_dict(self, read_sms_output):
        """ values to compare from the extractor """
        ser_dict = dict(read_sms_output.iloc[0])
        err_cols = {
            "confidence_level",
            "uncertainty",
            "lower_uncertainty",
            "upper_uncertainty",
        }
        for key in common_extractor_cols.union(err_cols):
            ser_dict.pop(key, None)
        return ser_dict

    @pytest.fixture(scope="class")
    def sm_dict(self, dummy_cat):
        """ values to compare from the arrivals """
        sm_dict = dict(dummy_cat[0].station_magnitudes[0].__dict__)
        # Remove unnecessary items
        err_objs = {"mag_errors"}
        for key in common_obj_attrs.union(err_objs):
            sm_dict.pop(key, None)
        # modify/add more complex items
        sm_dict["resource_id"] = sm_dict["resource_id"].id
        sm_dict["origin_id"] = sm_dict["origin_id"].id
        sm_dict["amplitude_id"] = sm_dict["amplitude_id"].id
        sm_dict["method_id"] = sm_dict["method_id"].id
        return sm_dict

    # general tests
    def test_type(self, read_sms_output):
        """ make sure a dataframe was returned """
        assert isinstance(read_sms_output, pd.DataFrame)

    def test_len(self, read_sms_output, dummy_cat):
        """ req_len should be the same as the sms req_len in events """
        req_len = len(dummy_cat[0].station_magnitudes) + len(
            dummy_cat[1].station_magnitudes
        )
        assert req_len == len(read_sms_output)

    def test_values(self, ser_dict, sm_dict):
        """ make sure the values of the first station magnitude are as expected """
        assert ser_dict == sm_dict

    # magnitude object tests
    def test_magnitude(self, dummy_mag):
        dummy_mag = dummy_mag.magnitudes[0]
        mag_df = station_magnitudes_to_df(dummy_mag)
        assert len(mag_df) == len(dummy_mag.station_magnitude_contributions)
        sm = mag_df.iloc[0]
        assert sm.magnitude_id == dummy_mag.resource_id.id

    # empty catalog tests
    def test_empty_catalog(self, empty_cat):
        """ ensure returns empty df with required columns """
        df = station_magnitudes_to_df(empty_cat)
        assert isinstance(df, pd.DataFrame)
        assert not len(df)
        assert set(df.columns).issubset(STATION_MAGNITUDE_COLUMNS)


class TestReadMagnitudes:
    # fixtures
    @pytest.fixture(scope="class")
    def dummy_cat(self):
        cat = ev.Catalog()
        eve = ev.Event()
        eve.origins.append(ev.Origin(time=UTCDateTime()))
        eve.magnitudes = mag_generator(["ML", "Md", "MW"])
        cat.append(eve)
        return cat

    @pytest.fixture(scope="class")
    def empty_cat(self):
        return ev.Catalog()

    @pytest.fixture(scope="class")
    def read_mags_output(self, dummy_cat):
        return magnitudes_to_df(dummy_cat)

    @pytest.fixture(scope="class")
    def ser_dict(self, read_mags_output):
        """ values to compare from the extractor """
        ser_dict = dict(read_mags_output.iloc[0])
        err_cols = {
            "confidence_level",
            "uncertainty",
            "lower_uncertainty",
            "upper_uncertainty",
        }
        for key in common_extractor_cols.union(err_cols):
            ser_dict.pop(key, None)
        return ser_dict

    @pytest.fixture(scope="class")
    def mag_dict(self, dummy_cat):
        """ values to compare from the arrivals """
        mag_dict = dict(dummy_cat[0].magnitudes[0].__dict__)
        # Remove unnecessary items
        extra_objs = {"mag_errors", "station_magnitude_contributions"}
        for key in common_obj_attrs.union(extra_objs):
            mag_dict.pop(key, None)
        # modify/add more complex items
        mag_dict["resource_id"] = mag_dict["resource_id"].id
        mag_dict["origin_id"] = mag_dict["origin_id"].id
        mag_dict["method_id"] = mag_dict["method_id"].id
        return mag_dict

    # general tests
    def test_type(self, read_mags_output):
        """ make sure a dataframe was returned """
        assert isinstance(read_mags_output, pd.DataFrame)

    def test_len(self, read_mags_output, dummy_cat):
        """ req_len should be the same as the amps req_len in events """
        req_len = len(dummy_cat[0].magnitudes)
        assert req_len == len(read_mags_output)

    def test_values(self, ser_dict, mag_dict):
        """ make sure the values of the first station magnitude are as expected """
        assert ser_dict == mag_dict

    # empty catalog tests
    def test_empty_catalog(self, empty_cat):
        """ ensure returns empty df with required columns """
        df = magnitudes_to_df(empty_cat)
        assert isinstance(df, pd.DataFrame)
        assert not len(df)
        assert set(df.columns).issubset(MAGNITUDE_COLUMNS)


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
