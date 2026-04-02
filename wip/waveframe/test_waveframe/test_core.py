"""
Tests for various core functionality of WaveFrame.
"""

import numpy as np

import pytest

from obsplus.waveframe.core import get_finite_segments


class TestGetFiniteSegments:
    """
    Test suite for indexing and handling NaN values in ndarrays.
    """

    def test_no_nan(self):
        """very simple test with No Nans"""
        ar = np.arange(100).reshape(10, 10)
        nns = get_finite_segments(ar)
        # no Nans so flat should be the same size
        assert nns.flat.size == ar.size
        for ind in nns.ind:
            # each segment should span exactly one row
            assert ind[0][1] - ind[0][0] == 1
            # and 10 columns
            assert ind[1][1] - ind[1][0] == 10

    def test_with_nan(self):
        """test with one NaN."""
        # get test data with one NaN in first row
        ar = np.arange(100).reshape(10, 10).astype(float)
        ar[0, 5] = np.NaN
        ar[-1, -1] = np.Inf
        nns = get_finite_segments(ar)
        assert nns.flat.size == ar.size - 2
        # there should now be 11 segments (2 from first row)
        assert len(nns.bp) == 11
        # first row should have two segments
        first, second = nns.ind[0], nns.ind[1]
        assert first[1][1] - first[1][0] <= 5
        assert second[1][1] - second[1][0] <= 5
        # last row should only be 9 long
        last = nns.ind[-1]
        assert last[1][1] - last[1][0] == 9

    def test_all_nan(self):
        """ensure all NaN raises."""
        ar = np.empty(100).reshape(10, 10) * np.NaN
        with pytest.raises(ValueError):
            get_finite_segments(ar)
