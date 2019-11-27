import pytest

from obsplus import WaveFrame
from obsplus.utils.testing import make_wf_with_nan


class TestStride:
    """ Tests for stridding data. """

    window_len = 1_500

    def test_overlap_gt_window_len_raises(self, waveframe_from_stream):
        """ Stride should rasie if the overlap is greater than window len. """
        wf = waveframe_from_stream
        with pytest.raises(ValueError):
            wf.stride(10, 100)

    def test_empty(self, waveframe_from_stream):
        """ Ensure striding works. """
        # Stridding with now input params should return a copy of waveframe.
        out = waveframe_from_stream.stride()
        assert isinstance(out, WaveFrame)
        assert out == waveframe_from_stream

    def test_overlap_default_window_len(self, waveframe_from_stream):
        """ Ensure strides can be overlapped. """
        wf = waveframe_from_stream
        # An overlap with the default window_len should also return a copy.
        wf2 = wf.stride(overlap=10)
        assert wf == wf2

    def test_no_overlap_half_len(self, waveframe_from_stream):
        """ ensure the stride when len is half creates a waveframe with 2x rows."""
        window_len = waveframe_from_stream.shape[-1] // 2
        wf = waveframe_from_stream
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

    def test_overlap_half_len(self, waveframe_from_stream):
        """ ensure half len """
        wf = waveframe_from_stream
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

    def test_drop_nan_column_all(self, waveframe_from_stream):
        """ Tests for dropping a column with all NaN. """
        wf = make_wf_with_nan(waveframe_from_stream, x_inds=0)
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

    def test_drop_nan_column_any(self, waveframe_from_stream):
        """ Tests for dropping a column with one NaN. """
        wf = make_wf_with_nan(waveframe_from_stream, 0, 0)
        # since only one value is NaN using how==all does nothing
        assert wf == wf.dropna(1, how="all")
        # but how==any should
        wf2 = wf.dropna(1, how="any")
        assert (wf["starttime"] < wf2["starttime"]).all()
        # the first index should always be 0
        assert wf2.data.columns[0] == 0

    def test_drop_nan_row_all(self, waveframe_from_stream):
        """ tests for dropping a row with all NaN"""
        wf = make_wf_with_nan(waveframe_from_stream, y_inds=0)
        wf2 = wf.dropna(0, how="all")
        assert wf2 == wf.dropna(0, how="any")
        # starttimes should not have changed
        assert (wf["starttime"][1:] == wf2["starttime"]).all()

    def test_drop_nan_row_any(self, waveframe_from_stream):
        """ test for dropping a row with one NaN. """
        wf = make_wf_with_nan(waveframe_from_stream, y_inds=0, x_inds=0)
        wf2 = wf.dropna(0, how="any")
        wf3 = wf.dropna(0, how="all")
        assert len(wf3) > len(wf2)

    def test_drop_all(self, waveframe_from_stream):
        """ tests for when all rows are dropped. """
        wf = make_wf_with_nan(waveframe_from_stream, x_inds=0)
        wf2 = wf.dropna(0, how="any")
        assert len(wf2) == 0


class TestResetIndex:
    """ tests for resetting index of waveframe. """


class TestSetIndex:
    """ Tests for setting index """
