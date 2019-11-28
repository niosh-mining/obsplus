"""
Waveframe core functionality and utilities.
"""
import abc
from typing import Callable, Dict

import numpy as np
import pandas as pd

from obsplus.constants import WAVEFRAME_STATS_DTYPES


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
        start_detla = time_delta * start_index
        df.loc[:, ("stats", "starttime")] = starttime + start_detla
        new_data_index = data.columns - start_index
        df = _reset_data_columns(df, new_data_index)
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


def _combine_stats_and_data(stats: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the data and stats dataframe into the waveframe input df.

    Parameters
    ----------
    stats
        A dataframe with the required stats columns.
    data
        A dataframe containing time-series.

    """
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

    @abc.abstractmethod
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """ default run is simply to return a copy. """

    def __call__(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Call the transformer, given the standard waveframe df.
        """
        # think about doing default (fast) validations and such here
        return self.method(self, df, *args, **kwargs)
