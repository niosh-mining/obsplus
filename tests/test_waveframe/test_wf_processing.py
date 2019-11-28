"""
Tests for processing methods of WaveFrame.
"""

import numpy as np
import pytest

from obsplus.utils.testing import make_wf_with_nan


@pytest.fixture()
def wf_with_offset(stream_wf):
    """ Create a waveframe with an offset. """
    # get waveframe with offset
    data = stream_wf.data
    index = data.columns.values.astype(int)
    array = data.values + 1
    array += index[np.newaxis, :]
    wf = stream_wf.from_data(array)
    return wf


class TestDetrend:
    """ Tests for applying detrend to waveframe. """

    @staticmethod
    def get_relative_offset(wf):
        """ Calculate mean / max of each trace, return np array. """
        data = wf.data
        vals = data.values
        mean = np.nanmean(vals, axis=1)
        maxval = np.nanmax(vals, axis=1)
        percent_offset = abs(mean / maxval)
        return percent_offset

    @pytest.fixture
    def wf_offset_nan(self, wf_with_offset):
        """ Add some NaN to offsets, just for fun. """
        return make_wf_with_nan(wf_with_offset, x_inds=(0, 100, 250, -1))

    def test_simple(self, wf_with_offset):
        """ ensure the simplest case works. """
        # get waveframe with offset
        out = wf_with_offset.detrend("simple")
        # calculate the mean, it can be slightly off 0 compared to max
        percent_offset = self.get_relative_offset(out)
        assert np.all(percent_offset < 0.1)

    def test_simple_with_nan(self, wf_offset_nan):
        """
        Make sure simple still works with NaNs
        """
        out = wf_offset_nan.detrend("simple")
        percent_offset = self.get_relative_offset(out)
        assert np.all([percent_offset < 0.1])

    def test_demean(self, wf_with_offset):
        """ tests for demean """
        out = wf_with_offset.detrend("demean")
        mean = out.data.mean(axis=1)
        assert np.allclose(mean, 0)
        # constant should do exactly the same thing
        assert wf_with_offset.detrend("constant") == out

    def test_demean_with_nan(self, wf_offset_nan):
        """ Make sure it still works with NaN. """
        out = wf_offset_nan.detrend("constant")
        mean = out.data.mean(axis=1)
        assert np.allclose(mean, 0)

    def test_detrend_linear(self, wf_with_offset):
        """ tests for linear detrending. """
        # out = wf_with_offset.detrend("linear")
        # breakpoint()
