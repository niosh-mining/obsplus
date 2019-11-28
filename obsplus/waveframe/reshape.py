"""
Waveframe logic for various reshaping and re-indexing.
"""

import numpy as np
import pandas as pd

from obsplus.waveframe.core import _combine_stats_and_data, DFTransformer


class _DataStridder(DFTransformer):
    """
    Class to encapsulate logic needed for stridding a waveframe df.

    This class should not be used directly but rather by the WaveFrame.
    """

    def _get_new_stats(self, start, y_inds, window_len, stats):
        """ get a new dataframe of stats. Also update start/end times. """
        # repeat stats rows for each entry in start
        out = stats.loc[y_inds].reset_index(drop=True)
        # get deltas to apply to starttimes. Accounts for new rows
        starttime_delta = np.tile(start, len(stats.index)) * out["delta"]
        # get deltas corresponding to endtimes
        endtime_delta = out["delta"] * (window_len - 1)
        out["starttime"] += starttime_delta
        out["endtime"] = out["starttime"] + endtime_delta
        return out

    def _get_data_array(self, start, end, window_len, data):
        """ create the array of data. """
        array = data.values
        # create empty NaN data array
        out = np.full((len(data) * len(start), window_len), np.NaN)
        count = 0
        for y_ind in data.index.values:
            for ind1, ind2 in zip(start, end):
                data_len = ind2 - ind1
                out[count, :data_len] = array[y_ind, ind1:ind2]
                count += 1
        return out

    def run(self, df, window_len=None, overlap=0) -> pd.DataFrame:
        """ Stride the dataframe, return new dataframe. """
        data, stats = df["data"], df["stats"]
        data_len = data.shape[-1]
        window_len, overlap = int(window_len or data_len), int(overlap)
        # if the stride length is the data length just return copy of df
        if window_len == data_len:
            return df.copy()
        if window_len < overlap:
            raise ValueError(f"window_len must be greater than overlap")
        # get start and stop indices
        start = np.arange(0, data_len, window_len - overlap)
        end = start + window_len
        end[end > data_len] = data_len
        # old y index for each stride length
        y_inds = np.repeat(stats.index.values, len(start))
        stats = self._get_new_stats(start, y_inds, window_len, stats)
        array = self._get_data_array(start, end, window_len, data)
        new_data = pd.DataFrame(array, index=stats.index)
        return _combine_stats_and_data(stats, new_data)
