"""
Tests for converting objects into times.
"""

from typing import Sequence

import numpy as np
import obspy
import pandas as pd
import pytest
from obspy import UTCDateTime

from obsplus.utils.time import to_datetime64, to_utc, to_timedelta64


class TestToNumpyDateTime:
    """ Tests for converting UTC-able objects to numpy datetime 64. """

    def test_simple(self):
        """ Test converting simple UTCDateTimable things """
        test_input = ("2019-01-10 11-12", obspy.UTCDateTime("2019-01-10T12-12"), 100)
        expected = np.array([obspy.UTCDateTime(x)._ns for x in test_input])
        out = np.array(to_datetime64(test_input)).astype(int)
        assert np.equal(expected, out).all()

    def test_with_nulls(self):
        """ Test for handling nulls. """
        test_input = (np.NaN, None, "", 15)
        out = np.array(to_datetime64(test_input))
        # first make sure empty values worked
        assert pd.isnull(out[:3]).all()
        assert out[-1].astype(int) == obspy.UTCDateTime(15)._ns

    def test_zero(self):
        """ Tests for input values as 0 or 0.0 """
        dt1 = to_datetime64(0)
        dt2 = to_datetime64(0.0)
        assert dt1.astype(int) == dt2.astype(int) == 0

    def test_npdatetime64_as_input(self):
        """ This should also work on np.datetime64. """
        test_input = np.array((np.datetime64(1000, "s"), np.datetime64(100, "ns")))
        out = to_datetime64(test_input)
        assert isinstance(out, np.ndarray)
        assert (test_input == out).all()

    def test_pandas_timestamp(self):
        """ Timestamps should also work. """
        kwargs = dict(year=2019, month=10, day=11, hour=12)
        ts = pd.Timestamp(**kwargs)
        out = to_datetime64((ts,))
        expected_out = (ts.to_datetime64(),)
        assert out == expected_out

    def test_utc_to_large(self):
        too_big = obspy.UTCDateTime("2600-01-01")
        with pytest.warns(UserWarning):
            out = to_datetime64(too_big)
        assert pd.Timestamp(out).year == 2262

    def test_series_to_datetimes(self):
        """ Series should be convertible to datetimes, but returns ndarray """
        ser = pd.Series([10, "2010-01-01"])
        out = to_datetime64(ser)
        assert isinstance(out, np.ndarray)


class TestTimeDelta:
    """ Tests for converting things to timedeltas. """

    def test_whole_number(self):
        """ test converting a number to a timedelta. """
        vals = [1, 2, 1000, 23, -122]
        out = [to_timedelta64(x) for x in vals]
        assert all(isinstance(x, np.timedelta64) for x in out)

    def test_float(self):
        """ Test converting floats to time deltas (interpreted as seconds)"""
        vals = [1.23322, 10.2323, -1232.22]
        out = [to_timedelta64(x) for x in vals]
        assert all([isinstance(x, np.timedelta64) for x in out])

    def test_series(self):
        """ Ensure an entire series can be converted to timedeltas."""
        ser = pd.Series([0, 2.22, 3, 5])
        out = to_timedelta64(ser)
        assert all([isinstance(x, np.timedelta64) for x in out])

    def test_array(self):
        """ Test the return values from an array. """
        ar = np.array([0, 2.22, 3, 5])
        out = to_timedelta64(ar)
        assert all([isinstance(x, np.timedelta64) for x in out])


class TestToUTC:
    """ Tests for converting things to UTCDateTime objects. """

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
        out = to_utc(value)
        # either a sequence or UTCDateTime should be returned
        assert isinstance(out, (Sequence, UTCDateTime, np.ndarray))
