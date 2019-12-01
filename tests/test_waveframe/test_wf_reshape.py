"""
Tests for manipulating shapes of waveframes.
"""
import numpy as np
import pandas as pd
import pytest

from obsplus import WaveFrame
from obsplus.utils.testing import make_wf_with_nan
from obsplus.utils.time import to_datetime64


class TestStride:
    """ Tests for stridding data. """

    window_len = 1_500

    def test_overlap_gt_window_len_raises(self, stream_wf):
        """ Stride should rasie if the overlap is greater than window len. """
        wf = stream_wf
        with pytest.raises(ValueError):
            wf.stride(10, 100)

    def test_empty(self, stream_wf):
        """ Ensure striding works. """
        # Stridding with now input params should return a copy of waveframe.
        out = stream_wf.stride()
        assert isinstance(out, WaveFrame)
        assert out == stream_wf

    def test_overlap_default_window_len(self, stream_wf):
        """ Ensure strides can be overlapped. """
        wf = stream_wf
        # An overlap with the default window_len should also return a copy.
        wf2 = wf.stride(overlap=10)
        assert wf == wf2

    def test_no_overlap_half_len(self, stream_wf):
        """ ensure the stride when len is half creates a waveframe with 2x rows."""
        window_len = stream_wf.shape[-1] // 2
        wf = stream_wf
        out = wf.stride(window_len=window_len).validate()
        assert len(out) == 2 * len(wf)
        assert out.shape[-1] == window_len
        # starttimes and endtime should have been updated
        starttimes, endtimes = out["starttime"], out["endtime"]
        delta = out["delta"]
        data_len = out.shape[-1]
        assert starttimes[0] + data_len * delta[0] == starttimes[1]
        # assert (endtimes - starttimes == (data_len - 1) * delta).all()
        assert endtimes[0] == starttimes[1] - delta[1]
        assert endtimes[0] + data_len * delta[0] == endtimes[1]

    def test_overlap_half_len(self, stream_wf):
        """ ensure half len """
        wf = stream_wf
        window_len = wf.shape[-1] // 2
        overlap = window_len // 2
        out = wf.stride(window_len=window_len, overlap=overlap).validate()
        data = out.data
        # no column should have all NaN values
        assert data.isnull().all(axis=1).sum() == 0
        # only one column for each channel should have NaN values
        assert data.isnull().any(axis=1).sum() == 3


class TestDropNa:
    """ tests for dropping null values. """

    def test_drop_nan_column_all(self, stream_wf):
        """ Tests for dropping a column with all NaN. """
        wf = make_wf_with_nan(stream_wf, x_inds=0)
        # first test drops based on rows, this should drop all rows
        wf2 = wf.dropna(1, how="any")
        assert wf2 is not wf
        # there should no longer be any NaN
        assert not wf2.data.isnull().any().any()
        # the start of the columns should be 0
        assert wf2.data.columns[0] == 0
        # the starttime should have been updated
        assert (wf2["starttime"] > wf["starttime"]).all()
        # dropping using the all keyword should also work
        assert wf.dropna(1, how="all") == wf2

    def test_drop_nan_column_any(self, stream_wf):
        """ Tests for dropping a column with one NaN. """
        wf = make_wf_with_nan(stream_wf, 0, 0)
        # since only one value is NaN using how==all does nothing
        assert wf == wf.dropna(1, how="all")
        # but how==any should
        wf2 = wf.dropna(1, how="any")
        assert (wf["starttime"] < wf2["starttime"]).all()
        # the first index should always be 0
        assert wf2.data.columns[0] == 0

    def test_drop_nan_row_all(self, stream_wf):
        """ tests for dropping a row with all NaN"""
        wf = make_wf_with_nan(stream_wf, y_inds=0)
        wf2 = wf.dropna(0, how="all")
        assert wf2 == wf.dropna(0, how="any")
        # starttimes should not have changed
        assert (wf["starttime"][1:] == wf2["starttime"]).all()

    def test_drop_nan_row_any(self, stream_wf):
        """ test for dropping a row with one NaN. """
        wf = make_wf_with_nan(stream_wf, y_inds=0, x_inds=0)
        wf2 = wf.dropna(0, how="any")
        wf3 = wf.dropna(0, how="all")
        assert len(wf3) > len(wf2)

    def test_drop_all(self, stream_wf):
        """ tests for when all rows are dropped. """
        wf = make_wf_with_nan(stream_wf, x_inds=0)
        wf2 = wf.dropna(0, how="any")
        assert len(wf2) == 0

    def test_drop_start_and_end(self, stream_wf):
        """ Drop a few samples from start and end, ensure times update."""
        wf = make_wf_with_nan(stream_wf, x_inds=(0, 1, -2, -1))
        delta = wf["delta"][0]
        start, end = wf["starttime"], wf["endtime"]
        # drop NaN, ensure start/end times are as expected.
        out = wf.dropna(axis=1, how="any")
        assert (out["starttime"] == (start + 2 * delta)).all()
        assert (out["endtime"] == (end - 2 * delta)).all()


class TestFillNaN:
    """ tests for filling NaN values. """

    def test_basic(self, stream_wf):
        """ Simple test for filling on NaN value. """
        wf = make_wf_with_nan(stream_wf, x_inds=0, y_inds=0)
        out = wf.fillna(2019)
        assert out.data.loc[0, 0] == 2019


class TestTrim:
    """ tests for trimming waveframes. """

    def test_trim_single_value(self, stream_wf):
        """ tests for trimming to a scalar value. """
        starttime = stream_wf["starttime"].iloc[0] + np.timedelta64(10, "s")
        endtime = stream_wf["endtime"].iloc[0] - np.timedelta64(10, "s")
        out = stream_wf.trim(starttime=starttime, endtime=endtime)
        new_starttime = out["starttime"]
        new_endtime = out["endtime"]
        assert (new_starttime >= starttime).all()
        assert (new_endtime <= endtime).all()

    def test_trim_out_of_existence(self, stream_wf):
        """ Tests for trimming out all data. """
        far_out = to_datetime64("2200-01-01")
        wf = stream_wf.trim(starttime=far_out)
        assert len(wf) == 0
        data, stats = wf.data, wf.stats
        assert len(data) == len(stats) == 0

    def test_trim_with_deltas(self, stream_wf):
        """ Tests for applying delta to stream_wf. """
        wf = stream_wf
        delta = np.timedelta64(5_000_001_000, "ns")
        start, end = wf["starttime"] + delta, wf["endtime"] - delta
        out = wf.trim(starttime=delta, endtime=-delta)
        assert (out["starttime"] >= start).all()
        assert (out["endtime"] <= end).all()
        # should return the same results as using absolute time
        assert out == stream_wf.trim(starttime=start, endtime=end)

    def test_trim_different_times(self, stream_wf):
        """ tests for different times on different chanenls. """
        wf = stream_wf
        deltas = wf["delta"] * np.arange(1, len(wf) + 1) * 10
        out = wf.trim(starttime=deltas, endtime=-deltas)
        # the first 10 values in row 1 should be 0, and first 20 in row 2
        assert pd.isnull(out.data.values[1, :10]).all()
        assert pd.isnull(out.data.values[2, :20]).all()
        # the starttimes should have changed slightly
        assert (out["starttime"] == (wf["starttime"] + out["delta"] * 10)).all()

    def test_trim_no_start(self, stream_wf):
        """ tests for trimming with no starttime. """
        center = stream_wf["starttime"] + np.timedelta64(15, "s")
        out = stream_wf.trim(endtime=center)
        assert (out["endtime"] <= center).all()
        # any difference between specified time and actual should be < 1 delta
        assert (abs(out["endtime"] - center) < out["delta"]).all()

    def test_trim_no_end(self, stream_wf):
        """ tests for trimming with no endtime. """
        center = stream_wf["starttime"] + np.timedelta64(15, "s")
        out = stream_wf.trim(starttime=center)
        assert (out["starttime"] >= center).all()
        # any difference between specified time and actual should be < 1 delta
        assert (abs(out["starttime"] - center) < out["delta"]).all()


class TestCutOut:
    """ tests for cutting out data from waveframe. """

    def test_basic(self, stream_wf):
        data, stats = stream_wf.data, stream_wf.stats
        start = stats["starttime"] + np.timedelta64(14, "s")
        end = stats["starttime"] + np.timedelta64(16, "s")
        delta = stats["delta"].iloc[0].to_timedelta64()
        t1, t2 = stats["starttime"].iloc[0], stats["endtime"].iloc[0]
        out = stream_wf.cutout(starttime=start, endtime=end)
        # the middle of data should have been removed, as well as end members
        cols = out.data.columns.values.astype(int)
        diffs = np.unique(cols[1:] - cols[:-1])
        assert len(diffs) == 2
        assert {1, 202} == set(list(diffs))
        removed_dates = np.arange(t1, t2, delta)
        # there should be no overlap
        assert not set(removed_dates) & set(out.data.index.values)


class TestResetIndex:
    """ tests for resetting index of waveframe. """


class TestSetIndex:
    """ Tests for setting index """
