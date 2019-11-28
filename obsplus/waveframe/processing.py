"""
Modules for performing simple processing with waveframes.
"""

import numpy as np
import pandas as pd
import scipy.signal

from obsplus.waveframe.core import DFTransformer, _combine_stats_and_data


class _WFDetrender(DFTransformer):
    """
    Logic for performing detrends on WaveFrame.
    """

    name = "detrend"

    # --- Supported methods

    def simple(self, df):
        """ detrending using a line drawn through the first and last point. """
        data, stats = df["data"], df["stats"]
        delta = stats["delta"]
        # get coefficients for forming line
        x0, x1 = self._make_first_last_time_values(df, data, stats)
        # get array of deltas
        data_inds = data.columns.values.astype(int) * delta[:, np.newaxis]
        # get offsets, subtract from current and return new df
        offsets = data_inds.astype(int) * x1 + x0
        new = data.values - offsets
        data_df = pd.DataFrame(new, index=data.index, columns=data.columns)
        return _combine_stats_and_data(stats=stats, data=data_df)

    def constant(self, df):
        """ Perform a simple demeaning on the data. """
        data, stats = df["data"], df["stats"]
        vals = data.values
        means = np.nanmean(vals, axis=1)
        new_values = vals - means[:, np.newaxis]
        new_data = pd.DataFrame(new_values, index=data.index, columns=data.columns)
        return _combine_stats_and_data(stats=stats, data=new_data)

    demean = constant

    def linear(self, df):
        """
        Perform linear detrending, possibly accounting for NaN values.
        """
        data, stats = df["data"], df["stats"]
        values = data.values
        finite = np.isfinite(values)
        # if everything is not NaN used fast path
        if np.all(finite):
            out = scipy.signal.detrend(values, axis=1)
        else:  # complicated logic to handle NaN
            out = self._linear_detrend_with_nan(values, finite)
        # create new df and return
        df = pd.DataFrame(out, index=data.index, columns=data.columns)
        return _combine_stats_and_data(stats=stats, data=df)

    # --- Helper functions

    def _linear_detrend_with_nan(self, values, finite):
        """ Apply linear detrend to data which have NaNs. """
        out = np.empty_like(values) * np.NaN
        vals = values[finite]  # flatten to only include non NaN
        # get indices of NoN Nan
        new_inds = np.indices(values.shape)
        row_ind = new_inds[0][finite]
        col_ind = new_inds[1][finite]
        # set breakpoints where new columns start
        bpoints = np.where(row_ind[:-1] < row_ind[1:])[0] + 1
        # apply piece-wise detrend
        detrended = scipy.signal.detrend(vals, type="linear", bp=bpoints)
        # indexes for flattened array (output of scipy)
        if_2 = np.concatenate([bpoints, [len(vals)]])
        if_1 = np.zeros_like(if_2)
        if_1[1:] = if_2[:-1]  # start is just the end of previous
        # indexes for original array (for putting data back in)
        col_start = col_ind[np.insert(bpoints, 0, 0)]  # start of col index
        row_ind = np.arange(len(if_1))  # row index
        # add detrended data back to array
        for fs, fe, row, col_s in zip(if_1, if_2, row_ind, col_start):
            ar = detrended[fs:fe]
            out[row, col_s : len(ar) + col_s] = ar
        return out

    def _make_linear_coefs(self, times: np.ndarray, vals: np.ndarray):
        """ return an array of x_0 and x_1 for each row in times"""
        assert times.shape == vals.shape

    def _make_first_last_time_values(self, df, data, stats):
        """ Get arrays of first and last non_NaN times and their values. """
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
