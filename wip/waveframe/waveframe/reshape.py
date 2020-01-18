"""
Waveframe logic for various reshaping and re-indexing.
"""
from contextlib import suppress
from typing import Union


import numpy as np
import pandas as pd

from obsplus.utils.time import to_datetime64, to_timedelta64

from obsplus.constants import time_types
from obsplus.waveframe.core import (
    _combine_stats_and_data,
    DFTransformer,
    _new_waveframe_df,
    get_time_array,
    _update_df_times,
)


def _get_absolute_time(
    time: Union[time_types, np.ndarray], ref_time: np.ndarray
) -> np.ndarray:
    """
    Get an absolute time from a possible reference time.

    Parameters
    ----------
    time
        Can either  be a an absolute time, or a timedelta with respect to
        ref_time.
    ref_time
        The object time is referenced to.
    """

    def _is_time_delta(obj):
        """ return True if an object is a timedelta like thing. """
        if isinstance(obj, (int, float)):
            return True
        dtype = getattr(obj, "dtype", None)
        if np.issubdtype(dtype, np.timedelta64):
            return True
        is_int = np.issubdtype(dtype, np.integer)
        is_float = np.issubdtype(dtype, np.floating)
        if is_int or is_float:
            return True
        return False

    # First try converting to datetime64, if that fails convert to timedelta.
    if _is_time_delta(time):
        dt = ref_time + to_timedelta64(time)
    else:
        dt = to_datetime64(time)
    return np.broadcast_to(dt, np.shape(ref_time))


class _NaNHandler(DFTransformer):
    """
    Class for handling NaNs.

    See `pandas.DataFrame.fillna` and `pandas.DataFrame.dropna`.
    """

    def fillna(self, df, *args, **kwargs):
        """ Fill data with NaN. """
        data = df.data.fillna(*args, **kwargs)
        return _new_waveframe_df(df, data=data)

    def dropna(self, df, *args, **kwargs):
        """ Drop data with NaN. """
        data = df.data.dropna(*args, **kwargs)
        df = _new_waveframe_df(df, data=data, allow_size_change=True)
        return _update_df_times(df, inplace=True)


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


class _Trimmer(DFTransformer):
    """
    Class for trimming and cutting out specified time periods.
    """

    def trim(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def cutout(self, *args, **kwargs):
        return self.run(*args, invert=True, **kwargs)

    def run(self, df, starttime=None, endtime=None, invert=False):
        """ Apply trimming to each row of waveframe's data. """
        data, stats = df["data"], df["stats"]
        # get current start and endtimes
        cstart, cend = stats["starttime"], stats["endtime"]
        t1 = _get_absolute_time(starttime, cstart)
        t2 = _get_absolute_time(endtime, cend)
        # get times corresponding to each sample and determine if in trim time
        time = get_time_array(df)
        to_trim = (time < t1[:, np.newaxis]) | (time > t2[:, np.newaxis])
        # invert (for cutout)
        if invert:
            to_trim = ~to_trim
        out = data.mask(to_trim).dropna(axis=1, how="all").dropna(axis=0, how="all")
        df = _combine_stats_and_data(data=out, stats=stats, allow_size_change=True)
        return _update_df_times(df, inplace=True)
