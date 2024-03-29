"""
Modules for performing simple processing with waveframes.
"""

import numpy as np
import pandas as pd
import scipy.signal

from obsplus.waveframe.core import (
    DFTransformer,
    _combine_stats_and_data,
    get_finite_segments,
    _new_waveframe_df,
)


class _WFDetrender(DFTransformer):
    """
    Logic for performing detrends on WaveFrame.
    """

    name = "detrend"

    # --- Supported methods

    def simple(self, df):
        """detrending using a line drawn through the first and last point."""
        data, stats = df["data"], df["stats"]
        delta = stats["delta"]
        # get coefficients for forming line
        x0, x1 = self._make_first_last_time_values(df, data, stats)
        # get array of deltas
        data_inds = data.columns.values.astype(int) * delta[:, np.newaxis]
        # get offsets, subtract from current and return new df
        offsets = data_inds.astype(int) * x1 + x0
        new = data.values - offsets
        return _new_waveframe_df(df, data=new)

    def constant(self, df):
        """A simple demeaning on the data via mean subtraction"""
        data = df.data.values
        means = np.nanmean(data, axis=1)
        out = data - means[:, np.newaxis]
        return _new_waveframe_df(df, data=out)

    demean = constant

    def linear(self, df, type="linear"):
        """
        Perform linear detrending, possibly accounting for NaN values.
        """
        data, stats = df["data"], df["stats"]
        values = data.values
        finite = np.isfinite(values)
        # if everything is not NaN used fast path
        if np.all(finite):
            out = scipy.signal.detrend(values, axis=1, type=type)
        else:  # complicated logic to handle NaN
            out = self._linear_detrend_with_nan(values, finite, method=type)
        # create new df and return
        df = pd.DataFrame(out, index=data.index, columns=data.columns)
        return _combine_stats_and_data(stats=stats, data=df)

    # --- Helper functions

    def _linear_detrend_with_nan(self, values, finite, method):
        """Apply linear detrend to data which have NaNs."""
        # init array for output
        out = values.copy()
        # get finite segments
        nns = get_finite_segments(values, finite=finite)
        detrended = scipy.signal.detrend(nns.flat, type=method, bp=nns.bp[:, 1])
        assert np.shape(detrended) == np.shape(nns.flat)
        # put humpty dumpty back together again
        for flat_ind, ind in zip(nns.bp, nns.ind):
            # get slice out of detrended result
            ar = detrended[flat_ind[0] : flat_ind[1]]
            out[ind[0][0] : ind[0][1], ind[1][0] : ind[1][1]] = ar
        return out

    def _make_linear_coefs(self, times: np.ndarray, vals: np.ndarray):
        """return an array of x_0 and x_1 for each row in times"""
        assert times.shape == vals.shape

    def _make_first_last_time_values(self, df, data, stats):
        """Get arrays of first and last non_NaN times and their values."""
        # get the last valid index
        data, stats = df["data"], df["stats"]
        first_ind = data.apply(axis=1, func=lambda x: x.first_valid_index())
        last_ind = data.apply(axis=1, func=lambda x: x.last_valid_index())
        # get times corresponding to first and last sample
        first_time = first_ind * stats["delta"] + stats["starttime"]
        last_time = last_ind * stats["delta"] + stats["starttime"]
        # get values for first and last sample
        first_amp = data.values[first_ind.index.values, first_ind.values]
        last_amp = data.values[last_ind.index.values, last_ind.values]
        # calculate coefficients, return
        dy = last_amp - first_ind
        dx = last_time - first_time
        return first_amp[:, np.newaxis], (dy / dx.astype(int))[:, np.newaxis]
