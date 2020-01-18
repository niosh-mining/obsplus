"""
Tests for processing methods of WaveFrame.
"""

import numpy as np
import pytest


from obsplus.waveframe.core import get_finite_segments

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

    nan_x_inds = (0, 100, 250, -1)

    @staticmethod
    def get_relative_offset(wf):
        """ Calculate mean / max of each trace, return np array. """
        data = wf.data
        vals = data.values
        mean = np.nanmean(vals, axis=1)
        maxval = np.nanmax(vals, axis=1)
        percent_offset = abs(mean / maxval)
        return percent_offset

    @staticmethod
    def assert_zeroish_mean_per_segment(wf):
        data = wf.data
        values = data.values
        finite_segs = get_finite_segments(values)
        for bp in finite_segs.bp:
            ar = finite_segs.flat[bp[0] : bp[1]]
            seg_mean = np.nanmean(ar)
            assert np.allclose(seg_mean, 0)

    @pytest.fixture
    def wf_offset_nan(self, wf_with_offset):
        """ Add some NaN to offsets, just for fun. """
        return make_wf_with_nan(wf_with_offset, x_inds=self.nan_x_inds)

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
        original_data = wf_with_offset.data.values.copy()
        out = wf_with_offset.detrend("linear")
        # the relative offsets should now be very close to 0
        rel_offset = self.get_relative_offset(out)
        assert np.allclose(rel_offset, 0)
        # the data should not have changed in place.
        assert np.allclose(original_data, wf_with_offset.data.values)
        # but should have changed on new object
        assert not np.allclose(original_data, out.data.values)
        # and all segments should have nearly 0 offsets
        self.assert_zeroish_mean_per_segment(out)

    def test_detrend_linear_with_nan(self, wf_offset_nan):
        """ test for linear detrending with NaN values. """
        original_data = wf_offset_nan.data.values
        original_copy = original_data.copy()
        out = wf_offset_nan.detrend("linear")
        # the data should not have changed in place.
        assert np.allclose(original_data, original_copy, equal_nan=True)
        # but should have changed
        assert not np.allclose(original_data, out.data.values)
        # the positions of NaNs should not have changed.
        nan1 = np.isnan(original_data)
        nan2 = np.isnan(out.data.values)
        assert np.all(nan1 == nan2)
        # and the relative offsets should be very close to 0
        rel_offsets = self.get_relative_offset(out)
        assert np.allclose(rel_offsets, 0)
        # and all segments should have nearly 0 offsets
        self.assert_zeroish_mean_per_segment(out)
