"""
Tests for converting objects into times.
"""

from typing import Sequence

import numpy as np
import obspy
import pandas as pd
import pytest
from obspy import UTCDateTime, Catalog
from obspy.core import event as ev
from obspy.core.event import Origin, Event

from obsplus import get_reference_time
from obsplus.utils.time import to_datetime64, to_utc, to_timedelta64


def append_func_name(list_obj):
    """decorator to append a function name to list_obj"""

    def wrap(func):
        list_obj.append(func.__name__)
        return func

    return wrap


class TestToNumpyDateTime:
    """Tests for converting UTC-able objects to numpy datetime 64."""

    def test_simple(self):
        """Test converting simple UTCDateTimable things"""
        test_input = ("2019-01-10 11-12", obspy.UTCDateTime("2019-01-10T12-12"), 100)
        expected_ns = np.array([obspy.UTCDateTime(x)._ns for x in test_input])
        dt64s = to_datetime64(test_input)
        new_ns = np.array(dt64s).astype(np.int64)
        assert np.equal(expected_ns, new_ns).all()

    def test_with_nulls(self):
        """Test for handling nulls."""
        test_input = (np.NaN, None, "", 15)
        out = np.array(to_datetime64(test_input))
        # first make sure empty values worked
        assert pd.isnull(out[:3]).all()
        assert out[-1].astype(np.int64) == obspy.UTCDateTime(15)._ns

    def test_zero(self):
        """Tests for input values as 0 or 0.0"""
        dt1 = to_datetime64(0)
        dt2 = to_datetime64(0.0)
        assert dt1.astype(np.int64) == dt2.astype(np.int64) == 0

    def test_npdatetime64_as_input(self):
        """This should also work on np.datetime64."""
        test_input = np.array((np.datetime64(1000, "s"), np.datetime64(100, "ns")))
        out = to_datetime64(test_input)
        assert isinstance(out, np.ndarray)
        assert (test_input == out).all()

    def test_pandas_timestamp(self):
        """Timestamps should also work."""
        kwargs = dict(year=2019, month=10, day=11, hour=12)
        ts = pd.Timestamp(**kwargs)
        out = to_datetime64((ts,))
        expected_out = (ts.to_datetime64(),)
        assert out == expected_out

    def test_utc_too_large(self):
        """Test a time larger than can fit into int64."""
        too_big = obspy.UTCDateTime("2600-01-01")
        with pytest.warns(UserWarning):
            out = to_datetime64(too_big)
        assert pd.Timestamp(out).year == 2262

    def test_npdatetime64_too_large(self):
        """Test np.datetime64s larger than can fit into int64"""
        too_big = np.array(
            [
                np.datetime64("2300-01-01"),
                np.datetime64("2020-01-01"),
                np.datetime64("2500-01-01"),
            ]
        )
        with pytest.warns(UserWarning):
            out = to_datetime64(too_big)
        years = out.astype("M8[Y]")
        assert np.array_equal(
            years,
            [
                np.datetime64("2262", "Y"),
                np.datetime64("2020", "Y"),
                np.datetime64("2262", "Y"),
            ],
        )

    def test_series_to_datetimes(self):
        """Series should be convertible to datetimes, return series"""
        ser = pd.Series([10, "2010-01-01"])
        out = to_datetime64(ser)
        assert isinstance(out, pd.Series)

    def test_nullish_values_returns_default(self):
        """Nullish values should return default values."""
        out1 = to_datetime64(None)
        assert pd.isnull(out1)
        assert out1 is not None

    def test_tuple_and_list(self):
        """tests for tuples and lists."""
        input1 = ["2020-01-03", obspy.UTCDateTime("2020-01-01").timestamp]
        out1 = to_datetime64(input1)
        out2 = to_datetime64(tuple(input1))
        assert np.all(out1 == out2)


class TestTimeDelta:
    """Tests for converting things to timedeltas."""

    def test_whole_number(self):
        """test converting a number to a timedelta."""
        vals = [1, 2, 1000, 23, -122]
        out = [to_timedelta64(x) for x in vals]
        assert all(isinstance(x, np.timedelta64) for x in out)

    def test_float(self):
        """Test converting floats to time deltas (interpreted as seconds)"""
        vals = [1.23322, 10.2323, -1232.22]
        out = [to_timedelta64(x) for x in vals]
        assert all([isinstance(x, np.timedelta64) for x in out])

    def test_series(self):
        """Ensure an entire series can be converted to timedeltas."""
        ser = pd.Series([0, 2.22, 3, 5])
        out = to_timedelta64(ser)
        assert all([isinstance(x, (np.timedelta64, pd.Timedelta)) for x in out])
        assert isinstance(out, pd.Series)

    def test_array(self):
        """Test the return values from an array."""
        ar = np.array([0, 2.22, 3, 5])
        out = to_timedelta64(ar)
        assert all([isinstance(x, np.timedelta64) for x in out])

    def test_no_precision_lost(self):
        """There should be no precision lost in converting to a timedelta."""
        td = np.timedelta64(1_111_111_111, "ns")
        out = to_timedelta64(td)
        assert out == td
        # and also in negative
        assert (-td) == to_timedelta64(-td)

    def test_identity_function_on_delta_array(self):
        """Delta array should simply return a delta array."""
        deltas = np.timedelta64(10_000_100, "us") * np.arange(10)
        out = to_timedelta64(deltas)
        assert np.all(deltas == out)

    def test_identity_function_on_delta_series(self):
        """Delta series should simply return an equal delta series."""
        deltas = np.timedelta64(10_000_100, "us") * np.arange(10)
        ser = pd.Series(deltas)
        out = to_timedelta64(ser)
        assert ser.equals(out)
        assert out is not ser

    def test_tuple_and_list(self):
        """tests for tuples and lists."""
        input1 = [2, -3, 4.5]
        out1 = to_timedelta64(input1)
        out2 = to_timedelta64(tuple(input1))
        assert np.all(out1 == out2)

    def test_nullish_values_returns_default(self):
        """Nullish values should return default values."""
        default = np.timedelta64(0, "s")
        out1 = to_timedelta64(None, default=default)
        assert out1 == default


class TestToUTC:
    """Tests for converting things to UTCDateTime objects."""

    # setup for test values
    utc1 = obspy.UTCDateTime("2019-01-10T12-12")
    utc_list = [utc1, utc1 + 2, utc1 + 3]
    dt64 = np.datetime64(1000, "ns")
    utc_able_list = [1, "2019-02-01", dt64]
    utc_values = [
        0,
        1_000_000,
        "2015-12-01",
        utc1,
        dt64,
        utc_list,
        np.array(utc_list),
        utc_able_list,
        np.array(utc_able_list, dtype=object),
        pd.Series(utc_able_list),
    ]

    @pytest.mark.parametrize("value", utc_values)
    def test_single_value(self, value):
        """Test either a sequence or UTCDateTime"""
        out = to_utc(value)
        assert isinstance(out, (Sequence, UTCDateTime, np.ndarray))


class TestGetReferenceTime:
    """tests for getting reference times from various objects"""

    time = obspy.UTCDateTime("2009-04-01")
    fixtures = []

    # fixtures
    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def utc_object(self):
        """Return a UTCDateTime object from reference time."""
        return obspy.UTCDateTime(self.time)

    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def timestamp(self):
        """Get the timestamp from reference time."""
        return self.time.timestamp

    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def event(self):
        """Create an event with an origin."""
        origin = Origin(time=self.time, latitude=47, longitude=-111.7)
        return Event(origins=[origin])

    @pytest.fixture(scope="class")
    @append_func_name(fixtures)
    def catalog(self, event):
        """Return a catalog from the event."""
        return Catalog(events=[event])

    @pytest.fixture(scope="class")
    def picks(self):
        """Create picks for testing."""
        t1, t2 = UTCDateTime("2016-01-01"), UTCDateTime("2015-01-01")
        picks = [ev.Pick(time=t1), ev.Pick(time=t2), ev.Pick()]
        return picks

    @pytest.fixture(scope="class")
    def event_only_picks(self, picks):
        """Return an event with only picks."""
        return ev.Event(picks=picks)

    @pytest.fixture(scope="class", params=fixtures)
    def time_outputs(self, request):
        """meta fixtures to gather up all the input types"""
        fixture_value = request.getfixturevalue(request.param)
        return get_reference_time(fixture_value)

    # tests
    def test_gather(self, utc_object, timestamp, event, catalog):
        """Simply gather aggregated fixtures so they are marked as used."""

    def test_is_utc_date(self, time_outputs):
        """ensure the output is a UTCDateTime"""
        assert isinstance(time_outputs, obspy.UTCDateTime)

    def test_time_equals(self, time_outputs):
        """ensure the outputs are equal to time on self"""
        assert time_outputs == self.time

    def test_empty_event_raises(self):
        """ensure an empty event will raise"""
        event = ev.Event()
        with pytest.raises(ValueError):
            get_reference_time(event)

    def test_event_with_picks(self, event_only_picks):
        """test that an event with picks, no origin, uses smallest pick"""
        t_expected = UTCDateTime("2015-01-01")
        t_out = get_reference_time(event_only_picks)
        assert t_expected == t_out

    def test_stream(self):
        """Ensure the start of the stream is returned."""
        st = obspy.read()
        out = get_reference_time(st)
        assert out == min([tr.stats.starttime for tr in st])

    def test_bad_type_raises(self):
        """Ensure a ValueError is raised when an unsupported type is used."""
        assert get_reference_time(None) is None
        with pytest.raises(TypeError):
            get_reference_time({})
