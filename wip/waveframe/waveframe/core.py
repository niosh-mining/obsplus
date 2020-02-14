"""
Waveframe core functionality and utilities.
"""
import abc
from collections import namedtuple
from typing import Optional, Callable, Dict

import numpy as np
import pandas as pd

import obsplus
from obsplus.constants import WAVEFRAME_STATS_DTYPES

FlatNan = namedtuple("FlatNan", "flat bp ind")


def get_finite_segments(
    array: np.ndarray, finite: Optional[np.ndarray] = None
) -> FlatNan:
    """
    Function to remove NaN from array and provide info on indices.

    Given an array is of dimension D with A non-nan contiguous segments along
    axis 1 the output is a named tuple with the following attributes:
        1. flat - flattened array with NaN removed
        2. bp - 2d array with start ar[0, :] and stop ar[1, :] indices
            indicating where each contiguous non NaN segments starts/ends in
            flattened array no NaN array.
        3. ind - array of shape (A, D, 2) where [a, :, 0] is the starting
            index corresponding to the start of segment a in arrays dimensions
            and [a, :, 1] is the ending index for the same segment.

    Notes
    -----
    This is only tested on a 2d arrays, it may not work on higher dimensions.

    Parameters
    ----------
    array
        Any numpy array which may have NaN values.
    finite
        A boolean array of the same shape as array indicating the finiteness
        of each element. If None it will be calculated.
    """

    def _get_break_points(ar_inds):
        """ Get the breakpoints from the flat array. """
        row_count = ar_inds[1]
        # we know a new segment starts if an expected value is skipped
        # this accounts for NaN interrupting segments or new rows
        _start = np.where(row_count[:-1] != (row_count[1:] - 1))[0] + 1
        start = np.insert(_start, 0, 0)
        end = np.append(start[1:], len(row_count))
        assert len(start) == len(end)
        return np.stack([start, end], axis=-1)

    def _get_unflat_indices(ar_inds, bps):
        """
        return an array of indices from original array corresponding
        to breakpoints.
        """
        start = np.array([x[bps[:, 0]] for x in ar_inds]).T
        stop = np.array([x[bps[:, 1] - 1] + 1 for x in ar_inds]).T
        return np.stack([start, stop], axis=-1)

    # Determine which values are finite and get indices of each value in array
    finite = np.isfinite(array) if finite is None else finite
    if not np.any(finite):
        msg = "Array with no finite values encountered"
        raise ValueError(msg)
    ar_index = np.indices(np.shape(array))
    # flatten the array and the indices of non-NaN values
    flat = array[finite]
    ar_inds = [x[finite] for x in ar_index]
    # first get points where new rows start
    bps = _get_break_points(ar_inds)
    old_inds = _get_unflat_indices(ar_inds, bps)
    return FlatNan(flat, bps, old_inds)


def _reset_data_columns(df, new_cols) -> pd.DataFrame:
    """
    Helper function for resetting the data columns of a waveframe df.

    Operates inplace to avoid copying data.
    """
    # convert current index to df, set new values and convert back
    ind_df = df.columns.to_frame()
    ind_df.loc[:, 1].loc["data"] = new_cols
    df.columns = pd.MultiIndex.from_frame(ind_df)
    return df


def _update_df_times(df, inplace=False) -> pd.DataFrame:
    """
    Update start/endtime based on the length of the data.

    As part of the process the columns of the data df will be reset to start
    at 0.
    """
    df = df if inplace else df.copy()
    time_delta = df.loc[:, ("stats", "delta")]
    starttime = df.loc[:, ("stats", "starttime")]
    time_index = df["data"].columns
    # update starttimes
    data = df.loc[:, "data"]
    start_index = data.columns[0]
    if start_index != 0:
        start_delta = time_delta * start_index
        df.loc[:, ("stats", "starttime")] = starttime + start_delta
        new_data_index = data.columns - start_index
        df = _reset_data_columns(df, new_data_index)
        time_index = df["data"].columns
    # update endtimes
    endtime = starttime + (time_index[-1] * time_delta)
    df.loc[:, ("stats", "endtime")] = endtime
    return df


def _enforce_monotonic_data_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    The index of the dataframe should be monotonic ints else re-index adding
    NaN for missing values.
    """
    columns = df["data"].columns
    # do nothing if the columns are already monotonic
    if columns.is_monotonic:
        return df
    # if not, re-index and re-create dataframe.
    new_index = np.arange(columns.min(), columns.max(), 1)
    new_data = df["data"].reindex(new_index, axis="columns")
    return _combine_stats_and_data(df["stats"], new_data)


def _new_waveframe_df(wdf, data=None, stats=None, allow_size_change=True):
    """
    Create a new waveframe df using the old one as a base.

    Parameters
    ----------
    wdf
        The old waveframe df.
    data
        If not None, data to used in new wdf.
    stats
        If not None, stats to use in new wdf.
    """

    def _df_from_array(old, ar):
        """ Create an dataframe from an array. """
        if isinstance(ar, pd.DataFrame):
            return ar
        index, cols = old.index, old.columns
        return pd.DataFrame(ar, index=index, columns=cols)

    wdf = wdf._df if isinstance(wdf, obsplus.WaveFrame) else wdf

    data_df = wdf["data"]
    stats_df = wdf["stats"]
    if data is not None:
        data_df = _df_from_array(data_df, data)
    if stats is not None:
        stats_df = _df_from_array(stats_df, stats)
    return _combine_stats_and_data(
        stats=stats_df, data=data_df, allow_size_change=allow_size_change
    )


def _combine_stats_and_data(
    stats: pd.DataFrame, data: pd.DataFrame, allow_size_change=False
) -> pd.DataFrame:
    """
    Combine the data and stats dataframe into the waveframe input df.

    Parameters
    ----------
    stats
        A dataframe with the required stats columns.
    data
        A dataframe containing time-series.
    allow_size_change
        If True, find the overlap in index and re-index both df.

    """
    # make sure data has at least one column (empty case)
    if not len(data.columns):
        data = pd.DataFrame(data, columns=[0])

    if allow_size_change:
        new_ind = stats.index.intersection(data.index)
        stats = stats.loc[new_ind]
        data = data.loc[new_ind]

    # simple checks
    assert set(WAVEFRAME_STATS_DTYPES).issubset(stats.columns)
    assert len(data) == len(stats)
    # concat together and add upper level index
    return pd.concat([stats, data], axis=1, keys=["stats", "data"])


def get_time_array(df):
    """ get an array of times corresponding to each sample in data. """
    data, stats = df["data"], df["stats"]
    start, delta = stats["starttime"], stats["delta"]
    data_inds = data.columns.values.astype(int) * delta[:, np.newaxis]
    datetimes = data_inds + start[:, np.newaxis]
    return datetimes


class DFTransformer:
    """
    A base class for implementing transformers for the df used by WaveFrame.
    """

    name = "base transformer"
    method_dict: Dict[str, Callable[..., pd.DataFrame]] = {}
    method: Callable[..., pd.DataFrame]

    def __init__(self, method="run"):
        """
        The init method is used only to set a specific method for transformation.

        Parameters
        ----------
        method
            The method name. The base class must implement a method with this
            name for the __call__ method to work.
        """
        # make sure there is an implementation for method else raise
        if method not in self.method_dict:
            msg = (
                f"{self.name} does not support method {method}. Supported "
                f"options are: {list(self.method_dict)}"
            )
            raise ValueError(msg)
        self.method = self.method_dict[method]

    def __init_subclass__(cls, **kwargs):
        """ create method dictionary """
        method_dict = {
            i: v
            for i, v in cls.__dict__.items()
            if not i.startswith("_") and callable(v)
        }
        cls.method_dict = method_dict

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """ default run is simply to return a copy. """
        return df

    def __call__(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Call the transformer, given the standard waveframe df.
        """
        # think about doing default (fast) validations and such here
        return self.method(self, df, *args, **kwargs)
