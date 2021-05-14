"""
Tests for converting catalogs to dataframes.
"""
from pathlib import Path

import numpy as np
import obspy
import obspy.core.event as ev
import pandas as pd
import pytest
from obspy import UTCDateTime

import obsplus
from obsplus import events_to_df, picks_to_df
from obsplus.constants import (
    EVENT_COLUMNS,
    PICK_COLUMNS,
    ARRIVAL_COLUMNS,
    AMPLITUDE_COLUMNS,
    STATION_MAGNITUDE_COLUMNS,
    MAGNITUDE_COLUMNS,
    EVENT_DTYPES,
)
from obsplus.events.pd import (
    amplitudes_to_dataframe,
    station_magnitudes_to_dataframe,
    arrivals_to_dataframe,
    picks_to_dataframe,
    event_to_dataframe,
    magnitudes_to_dataframe,
)
from obsplus.utils.misc import register_func
from obsplus.utils.time import to_datetime64, to_utc

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


def pick_generator(scnls):
    """Create picks for testing."""
    picks = []
    for scnl in scnls:
        p = ev.Pick(
            time=UTCDateTime(), waveform_id=ev.WaveformStreamID(seed_string=scnl)
        )
        picks.append(p)
    return picks


def make_arrivals(picks):
    """Create arrivals for testing."""
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


def make_amplitudes(scnls=None, picks=None):
    """Create amplitudes for testing."""
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
    """ Function to create station magntiudes for testing."""
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
    """Function to create magnitudes for testing."""
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


def floatify_dict(some_dict):
    """
    Iterate a dict and convert all TimeStamps/datetime64 to floats.
    Then round all floats to nearest 4 decimals.
    """
    out = {}
    for i, v in some_dict.items():
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            v = to_utc(v).timestamp
        if isinstance(v, float):
            v = np.round(v, 4)
        out[i] = v
    return out


@pytest.fixture
def event_directory():
    """ Return the directory of the bingham_test catalog. """
    ds = obsplus.load_dataset("bingham_test")
    return ds.event_path


@pytest.fixture
def event_file(event_directory):
    """ Return a path to the first event file in the bingham_test bank. """
    first = list(Path(event_directory).rglob("*.xml"))[0]
    return first


@pytest.fixture(scope="class")
def empty_catalog():
    """ run an empty events through the picks_to_df function """
    event = ev.Event()
    cat = obspy.Catalog(events=[event])
    return picks_to_dataframe(cat)


# --------------- tests


class TestCat2Df:
    """ test the simplified dataframe conversion """

    required_columns = EVENT_COLUMNS
    fixtures = []

    # fixtures
    @pytest.fixture(scope="class")
    @register_func(fixtures)
    def events_from_catalog(self):
        """ read events from a events object """
        cat = obspy.read_events()
        return event_to_dataframe(cat)

    @pytest.fixture(scope="class")
    @register_func(fixtures)
    def events_from_dataframe(self):
        """ Create events df from an event dataframe."""
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
    @register_func(fixtures)
    def catalog_no_magnitude(self):
        """ get a events with no magnitudes (should just fill with NaN) """
        t1 = obspy.UTCDateTime("2099-04-01T00-01-00")
        ori = ev.Origin(time=t1, latitude=47.1, longitude=-100.22)
        event = ev.Event(origins=[ori])
        cat = ev.Catalog(events=[event])
        return events_to_df(cat)

    @pytest.fixture(scope="class")
    @register_func(fixtures)
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
        for event in cat:
            for pick in event.picks:
                pick.evaluation_status = "rejected"
        return cat

    @pytest.fixture
    def events_with_origin_quality(self, bingham_dataset):
        """ Return events with an overridden origin quality """
        cat = bingham_dataset.event_client.get_events().copy()
        oq = ev.OriginQuality(
            associated_phase_count=42,
            used_phase_count=10,
            associated_station_count=21,
            used_station_count=5,
        )
        for event in cat:
            origin = event.preferred_origin()
            origin.quality = oq
        return cat

    @pytest.fixture(scope="class")
    def df(self, test_catalog):
        """ call the catalog2df method, return result"""
        cat = test_catalog.copy()
        return event_to_dataframe(cat)

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
                ed = ""
            assert ed == description

    def test_str(self):
        """ ensure there is a string rep for events_to_df. """
        cat_to_df_str = str(events_to_df)
        assert isinstance(cat_to_df_str, str)  # dumb test to boost coverage

    def test_event_id_in_columns(self, df):
        """
        Sometimes the event_id gets changed to the index, make sure it is
        still a column.
        """
        cols = df.columns
        assert "event_id" in cols

    def test_column_datatypes(self, df):
        """ Ensure the expected columns are numpy datetime objects. """
        dtypes = dict(EVENT_DTYPES)
        dtypes.pop("longitude")
        expected = df.astype(dtypes).dtypes
        assert (expected == df.dtypes).all()

    def test_basics(self, read_events_output):
        """ make sure a dataframe is returned """
        assert isinstance(read_events_output, pd.DataFrame)
        assert len(read_events_output)

    def test_rejected_phases_still_counted(self, events_rejected_picks):
        """ ensure rejected picks are still counted in arrival numbering """
        df = events_to_df(events_rejected_picks)
        assert (df.p_phase_count != 0).all()

    def test_origin_quality_wins(self, events_with_origin_quality):
        """
        Ensure the phase counts from the OriginQuality take priority over a
        count of picks/arrivals
        """
        df = events_to_df(events_with_origin_quality)
        assert df.associated_phase_count.iloc[0] == 42
        assert df.used_phase_count.iloc[0] == 10

    def test_event_bank_to_df(self, default_ebank):
        """ Ensure event banks can be used to get dataframes. """
        df = obsplus.events_to_df(default_ebank)
        assert isinstance(df, pd.DataFrame)

    def test_event_directory_to_df(self, event_directory):
        """Test for getting dataframe from event directory."""
        df = events_to_df(event_directory)
        assert len(df)
        assert isinstance(df, pd.DataFrame)

    def test_cat_to_df_method(self):
        """ ensure the events object has the to_df method bolted on """
        cat = obspy.read_events()
        df = cat.to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(cat)

    def test_gather(
        self,
        events_from_catalog,
        events_from_dataframe,
        catalog_no_magnitude,
        catalog_empty,
    ):
        """
        This test simply gathers up all the aggregated fixtures so they
        are properly marked as used.
        """


class TestLongitude:
    """
    Tests for ensuring longitudes get mapped to a domain of -180 to 180
    as expected.
    """

    def test_catalog(self):
        """Tests for (modified) default catalog."""
        import obspy
        import obsplus

        cat = obspy.read_events()
        cat[0].origins[0].longitude = 799
        cat[1].origins[0].longitude = -181

        df = obsplus.events_to_df(cat)
        longitudes = df["longitude"]
        assert np.all(np.abs(longitudes) <= 180)


class TestCat2DfPreferredThings:
    """ Make sure the preferred origins/mags show up in the df. """

    # fixtures
    @pytest.fixture(scope="class", autouse=True)
    def preferred_magnitudes(self, test_catalog):
        """
        Set the preferred magnitudes to the first magnitudes,
        return list of magnitudes.
        """
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
        """
        Set the preferred magnitudes to the first magnitudes,
        return list of magnitudes.
        """
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
            assert np.allclose(origin.latitude, row.latitude)
            assert np.allclose(origin.longitude, row.longitude)
            assert origin.time == obspy.UTCDateTime(row.time)

    def test_magnitudes(self, df, preferred_magnitudes, test_catalog):
        """ ensure the origins are correct """
        for ind, row in df.iterrows():
            mag = preferred_magnitudes[ind]
            assert mag.mag == row.magnitude
            mtype1 = str(row.magnitude_type).upper()
            mtype2 = str(mag.magnitude_type or "").upper()
            assert mtype1 == mtype2


class TestReadPicks:
    """ ensure phase picks can be read in """

    fixtures = []
    cat_name = "2016-04-13T23-51-13.xml"

    @pytest.fixture(scope="class")
    def tcat(self, catalog_cache):
        """ read in an event for testing """
        return catalog_cache[self.cat_name]

    @pytest.fixture(scope="class")
    @register_func(fixtures)
    def catalog_output(self, tcat):
        """ call read_picks on the events, return result """
        return picks_to_dataframe(tcat)

    @pytest.fixture(scope="class")
    @register_func(fixtures)
    def dataframe_output(self, tcat):
        """ return read_picks result from reading dataframe """
        df = picks_to_df(tcat)
        return picks_to_dataframe(df)

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
        return picks_to_dataframe(ev.Event(picks=picks))

    @pytest.fixture(scope="class", params=fixtures)
    def read_picks_output(self, request):
        """ return the outputs from the fixtures """
        return request.getfixturevalue(request.param)

    @pytest.fixture
    def bingham_cat_only_picks(self, bingham_dataset):
        """ return bingham_test catalog with everything but picks removed """
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
        assert (df["event_time"] == df["time"].min()).all()

    def test_unique_event_time_no_origin(self, bingham_cat_only_picks):
        """ Ensure events with no origin don't all return the same time. """
        df = picks_to_dataframe(bingham_cat_only_picks)
        assert len(df["event_time"].unique()) == len(df["event_id"].unique())

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
        df = picks_to_dataframe(pick)
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
        df = picks_to_dataframe(pick)
        assert df.onset.iloc[0] == ""
        assert df.polarity.iloc[0] == ""

    def test_from_file(self, event_file):
        """ test for reading phase picks from files. """
        df = picks_to_df(event_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df)

    def test_from_event_bank(self, default_ebank):
        """ Ensure event banks can be used to get dataframes. """
        df = picks_to_df(default_ebank)
        assert isinstance(df, pd.DataFrame)

    def test_from_event_directory(self, event_directory):
        """Test extracting info from an event directory."""
        df = picks_to_df(event_directory)
        assert len(df)
        assert isinstance(df, pd.DataFrame)

    def test_from_dataframe_int_location_codes(self, bingham_dataset):
        """
        Tests for a dataframe with an int column defining location code.
        These can drop necessary 0s.
        """
        events = bingham_dataset.event_client.get_events()
        df1 = obsplus.picks_to_df(events)
        df = df1.copy()
        df["location"] = df["location"].astype(int)
        df2 = picks_to_df(df)
        assert (df2["location"] == df1["location"]).all()
        # ensure the seed ids are still consistent
        seed_id_loc = df2["seed_id"].str.split(".", expand=True)[2]
        assert (seed_id_loc == df2["location"]).all()

    def test_picks_one_digit_location(self):
        """
        Ensure 1 digit location codes get preserved
        """
        picks = pick_generator(["UU.TMU.1.HHZ", "UU.TMU.01.HHZ"])
        df = picks_to_df(picks)
        # there should be two unique location code and seed_ids.
        assert len(set(df["location"])) == 2
        assert len(set(df["seed_id"])) == 2

    def test_dot_in_location_code(self):
        """Ensure a dot in the location code causes a ValueError. """
        waveform_id = ev.WaveformStreamID(
            network_code="UU",
            station_code="TMU",
            location_code="1.0",
            channel_code="HHZ",
        )
        pick = ev.Pick(time=obspy.UTCDateTime("2020-01-01"), waveform_id=waveform_id)
        with pytest.raises(ValueError):
            _ = obsplus.picks_to_df([pick])

    def test_gather(self, catalog_output, dataframe_output):
        """ Simply gather aggregated fixtures so they are marked as used. """


class TestReadArrivals:
    """Test case for reading arrivals into a dataframe."""

    # fixtures
    @pytest.fixture(scope="class")
    def dummy_cat(self):
        """Return a dummy catalog for testing arrival reading."""
        scnls1 = ["UK.STA1..HHZ", "UK.STA2..HHZ"]
        cat = ev.Catalog()
        eve1 = ev.Event()
        eve1.origins.append(ev.Origin(time=UTCDateTime()))
        eve1.preferred_origin_id = eve1.origins[0].resource_id
        picks = pick_generator(scnls1)
        eve1.picks = picks
        eve1.preferred_origin().arrivals = make_arrivals(picks)
        scnls2 = ["UK.STA3..HHZ", "UK.STA4..HHZ", "UK.STA5..HHZ"]
        eve2 = ev.Event()
        eve2.origins.append(ev.Origin(time=UTCDateTime()))
        eve2.preferred_origin_id = eve2.origins[0].resource_id
        picks = pick_generator(scnls2)
        eve2.picks = picks
        eve2.preferred_origin().arrivals = make_arrivals(picks)
        cat.events = [eve1, eve2]
        return cat

    @pytest.fixture(scope="class")
    def no_origin(self):
        """Return a catalog with no origin."""
        cat = ev.Catalog()
        cat.append(ev.Event())
        return cat

    @pytest.fixture(scope="class")
    def read_arr_output(self, dummy_cat):
        """Read the dummy_catalog into an arrival dataframe."""
        return arrivals_to_dataframe(dummy_cat)

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
        floatify_dict(ser_dict)
        assert floatify_dict(ser_dict) == floatify_dict(arr_dict)

    # empty catalog tests
    def test_empty_catalog(self):
        """ ensure returns empty df with required columns """
        empty_cat = ev.Catalog()
        df = arrivals_to_dataframe(empty_cat)
        assert isinstance(df, pd.DataFrame)
        assert not len(df)
        assert set(df.columns).issubset(ARRIVAL_COLUMNS)

    def test_no_origin(self, no_origin):
        """ ensure returns empty df with required columns """
        df = arrivals_to_dataframe(no_origin)
        assert isinstance(df, pd.DataFrame)
        assert not len(df)
        assert set(df.columns).issubset(ARRIVAL_COLUMNS)

    def test_from_file(self, event_file):
        """ test for reading arrivals from files. """
        df = arrivals_to_dataframe(event_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df)

    def test_from_event_bank(self, default_ebank):
        """ Ensure event banks can be used to get dataframes. """
        df = arrivals_to_dataframe(default_ebank)
        assert isinstance(df, pd.DataFrame)

    def test_from_event_directory(self, event_directory):
        """Test for extracting info from a directory of events."""
        df = arrivals_to_dataframe(event_directory)
        assert len(df)
        assert isinstance(df, pd.DataFrame)

    def test_arrivals_from_origin(self, dummy_cat):
        """ tests for specifically extracting arrivals from origin objects """
        origin = dummy_cat[0].origins[0]
        df = arrivals_to_dataframe(origin)
        assert len(df) == len(origin.arrivals)


class TestReadAmplitudes:
    """Tests for reading amplitudes into dataframes."""

    # fixtures
    @pytest.fixture(scope="class")
    def dummy_cat(self):
        """Return a dummy catalog for testing."""
        scnls1 = ["UK.STA1..HHZ", "UK.STA2..HHZ"]
        cat = ev.Catalog()
        eve1 = ev.Event()
        eve1.origins.append(ev.Origin(time=UTCDateTime()))
        eve1.picks = pick_generator(scnls1)
        eve1.amplitudes = make_amplitudes(picks=eve1.picks)
        scnls2 = ["UK.STA3..HHZ", "UK.STA4..HHZ", "UK.STA5..HHZ"]
        eve2 = ev.Event()
        eve2.origins.append(ev.Origin(time=UTCDateTime()))
        eve2.amplitudes = make_amplitudes(scnls=scnls2)
        cat.events = [eve1, eve2]
        return cat

    @pytest.fixture(scope="class")
    def read_amps_output(self, dummy_cat):
        """Convert the amplitudes to dataframes."""
        return amplitudes_to_dataframe(dummy_cat)

    @pytest.fixture(scope="class")
    def amplitude(self, dummy_cat):
        """Return the first amplitude."""
        return dummy_cat[0].amplitudes[0]

    @pytest.fixture(scope="class")
    def amp_series(self, read_amps_output):
        """Return an amplitude series."""
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
        assert floatify_dict(ser_dict) == floatify_dict(amp_dict)

    def test_creation_time(self, amplitude, amp_series):
        """Ensure creation time was included."""
        assert amp_series["creation_time"] == to_datetime64(
            amplitude.creation_info.creation_time
        )
        assert amp_series["author"] == amplitude.creation_info.author
        assert amp_series["agency_id"] == amplitude.creation_info.agency_id

    def test_empty_catalog(self):
        """ ensure returns empty df with required columns """
        df = amplitudes_to_dataframe(ev.Catalog())
        assert isinstance(df, pd.DataFrame)
        assert not len(df)
        assert set(df.columns).issubset(AMPLITUDE_COLUMNS)

    def test_from_file(self, event_file):
        """ test for reading amplitudes from files. """
        df = amplitudes_to_dataframe(event_file)
        assert isinstance(df, pd.DataFrame)

    def test_from_event_bank(self, default_ebank):
        """ Ensure event banks can be used to get dataframes. """
        df = amplitudes_to_dataframe(default_ebank)
        assert isinstance(df, pd.DataFrame)

    def test_from_event_directory(self, event_directory):
        """Test extracting info from a test directory."""
        df = amplitudes_to_dataframe(event_directory)
        assert isinstance(df, pd.DataFrame)


class TestReadStationMagnitudes:
    """Tests for reading station magnitude info into a dataframe."""

    # fixtures
    @pytest.fixture(scope="class")
    def dummy_cat(self):
        """ Return a dummy catalog for testing."""
        scnls1 = ["UK.STA1..HHZ", "UK.STA2..HHZ"]
        cat = ev.Catalog()
        eve1 = ev.Event()
        eve1.origins.append(ev.Origin(time=UTCDateTime()))
        eve1.amplitudes = make_amplitudes(scnls1)
        eve1.station_magnitudes = sm_generator(amplitudes=eve1.amplitudes)
        scnls2 = ["UK.STA3..HHZ", "UK.STA4..HHZ", "UK.STA5..HHZ"]
        eve2 = ev.Event()
        eve2.origins.append(ev.Origin(time=UTCDateTime()))
        eve2.station_magnitudes = sm_generator(scnls=scnls2)
        cat.events = [eve1, eve2]
        return cat

    @pytest.fixture(scope="class")
    def dummy_mag(self):
        """Return a dummy magnitude."""
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
    def read_sms_output(self, dummy_cat):
        """Read station magnitudes from the dummy catalog."""
        return station_magnitudes_to_dataframe(dummy_cat)

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
        assert floatify_dict(ser_dict) == floatify_dict(sm_dict)

    # magnitude object tests
    def test_magnitude(self, dummy_mag):
        """Test getting info from magnitudes."""
        dummy_mag = dummy_mag.magnitudes[0]
        mag_df = station_magnitudes_to_dataframe(dummy_mag)
        assert len(mag_df) == len(dummy_mag.station_magnitude_contributions)
        sm = mag_df.iloc[0]
        assert sm.magnitude_id == dummy_mag.resource_id.id

    # empty catalog tests
    def test_empty_catalog(self):
        """ ensure returns empty df with required columns """
        empty_cat = ev.Catalog()
        df = station_magnitudes_to_dataframe(empty_cat)
        assert isinstance(df, pd.DataFrame)
        assert not len(df)
        assert set(df.columns).issubset(STATION_MAGNITUDE_COLUMNS)

    def test_from_file(self, event_file):
        """ test for reading station magnitudes from files. """
        df = station_magnitudes_to_dataframe(event_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df)

    def test_from_event_bank(self, default_ebank):
        """ Ensure event banks can be used to get dataframes. """
        df = station_magnitudes_to_dataframe(default_ebank)
        assert isinstance(df, pd.DataFrame)

    def test_from_event_directory(self, event_directory):
        """Test extracting info from a directory of events."""
        df = station_magnitudes_to_dataframe(event_directory)
        assert len(df)
        assert isinstance(df, pd.DataFrame)


class TestReadMagnitudes:
    """Tests for getting magnitude info into a dataframe."""

    # fixtures
    @pytest.fixture(scope="class")
    def dummy_cat(self):
        """Return a dummy catalog for testing."""
        cat = ev.Catalog()
        eve = ev.Event()
        eve.origins.append(ev.Origin(time=UTCDateTime()))
        eve.magnitudes = mag_generator(["ML", "Md", "MW"])
        cat.append(eve)
        return cat

    @pytest.fixture(scope="class")
    def read_mags_output(self, dummy_cat):
        """Convert dummy catalog to magnitude df."""
        return magnitudes_to_dataframe(dummy_cat)

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
        assert floatify_dict(ser_dict) == floatify_dict(mag_dict)

    # empty catalog tests
    def test_empty_catalog(self):
        """ ensure returns empty df with required columns """
        empty_cat = ev.Catalog()
        df = magnitudes_to_dataframe(empty_cat)
        assert isinstance(df, pd.DataFrame)
        assert not len(df)
        assert set(df.columns).issubset(MAGNITUDE_COLUMNS)

    def test_event_bank_to_df(self, default_ebank):
        """ Ensure event banks can be used to get dataframes. """
        df = obsplus.magnitudes_to_df(default_ebank)
        assert isinstance(df, pd.DataFrame)
        assert len(df)

    def test_from_file(self, event_file):
        """ Test for reading magnitudes from files. """
        df = magnitudes_to_dataframe(event_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df)

    def test_from_event_bank(self, default_ebank):
        """ Ensure event banks can be used to get dataframes. """
        df = magnitudes_to_dataframe(default_ebank)
        assert isinstance(df, pd.DataFrame)

    def test_from_event_directory(self, event_directory):
        """Tests read magnitudes on event directory."""
        df = magnitudes_to_dataframe(event_directory)
        assert len(df)
        assert isinstance(df, pd.DataFrame)


class TestReadBingham:
    """ Test for reading a variety of pick formats from the bingham_test dataset. """

    event_fixtures = []
    picks_fixtures = []

    @pytest.fixture(scope="class")
    @register_func(event_fixtures)
    @register_func(picks_fixtures)
    def catalog(self, bingham_dataset):
        """ return the bingham_test catalog """
        return bingham_dataset.event_client.get_events()

    @pytest.fixture(scope="class")
    @register_func(picks_fixtures)
    def picks_list(self, catalog):
        """ Get the picks from the catalog. """
        return [pick for eve in catalog for pick in eve.picks]

    @pytest.fixture(scope="class")
    @register_func(event_fixtures)
    def catalog_df(self, catalog):
        """ Get a catalog of events. """
        return events_to_df(catalog)

    @pytest.fixture(scope="class")
    @register_func(event_fixtures)
    def event_csv_path(self, catalog_df, tmp_path_factory):
        """ Return a path to a saved csv of event files. """
        tmp_path = Path(tmp_path_factory.mktemp("ecsv")) / "events.csv"
        catalog_df.to_csv(tmp_path)
        return tmp_path

    @pytest.fixture(scope="class")
    @register_func(picks_fixtures)
    def pick_dataframe(self, catalog):
        """ return a dataframe of picks. """
        return picks_to_df(catalog)

    @pytest.fixture(scope="class")
    @register_func(picks_fixtures)
    def pick_csv_path(self, pick_dataframe, tmp_path_factory):
        """ Return a path to a pick dataframe. """
        tmp_path = Path(tmp_path_factory.mktemp("pickcsv")) / "picks.csv"
        pick_dataframe.to_csv(tmp_path)
        return tmp_path

    # --- aggregation fixtures

    @pytest.fixture(scope="class", params=event_fixtures)
    def event_df(self, request):
        """Collect all the supported inputs and parametrize."""
        return events_to_df(request.getfixturevalue(request.param))

    @pytest.fixture(scope="class", params=picks_fixtures)
    def pick_df(self, request):
        """Collect everything in the pick dataframe."""
        df = picks_to_df(request.getfixturevalue(request.param))
        return df

    # --- tests

    def test_event_df_len(self, event_df, catalog):
        """Ensure the length of the event_df and catalog are equal."""
        assert len(event_df) == len(catalog)

    def test_event_column_order(self, event_df):
        """Ensure the order of the columns is correct."""
        cols = list(event_df.columns)
        assert list(EVENT_COLUMNS) == cols[: len(EVENT_COLUMNS)]

    def test_pick_len(self, pick_df, picks_list):
        """Ensure the correct number of items was returned."""
        assert len(pick_df) == len(picks_list)

    def test_pick_column_order(self, pick_df):
        """Ensure the order of the columns is correct."""
        cols = list(pick_df.columns)
        assert list(PICK_COLUMNS) == cols[: len(PICK_COLUMNS)]

    def test_event_id(self, pick_df):
        """
        There can either be all Nan in the event_id column (for the case of the
        dataframe from picks list) or all not NaN for other cases.
        """
        null_count = pick_df["event_id"].isnull().sum()
        assert null_count == 0 or null_count == len(pick_df)
