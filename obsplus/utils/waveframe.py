"""
Waveframe specific utilities.
"""
from typing import List

import numpy as np
import obspy
import pandas as pd

from obsplus.constants import BULK_WAVEFORM_COLUMNS, WAVEFRAME_STATS_DTYPES
from obsplus.constants import TIME_COLUMNS
from obsplus.constants import WaveformClient, bulk_waveform_arg_type
from obsplus.utils.pd import (
    apply_funcs_to_columns,
    order_columns,
    cast_dtypes,
    get_waveforms_bulk_args,
)
from obsplus.utils.time import to_utc
from obsplus.utils.waveforms import get_waveform_client, stream_bulk_split


# functions to be applied to stats after converting to df
# STATS_FUNCS = {"delta": to_datetime64, "starttime": to_utc, "endtime": to_utc}


class DfPartDescriptor:
    """
    A simple descriptor granting access to various parts of a dataframe.
    """

    def __init__(self, df_name):
        self._df_name = df_name

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        return getattr(instance, self._df_name)[self._name]


def _time_to_utc(df: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """
    Convert time columns in a pandas dataframe to UTCDateTime objects.
    """
    df = df.copy() if inplace else df
    col_funcs = {name: to_utc for name in TIME_COLUMNS}
    return apply_funcs_to_columns(df, col_funcs)


def _get_waveforms_bulk(
    waveform_client: WaveformClient, bulk: bulk_waveform_arg_type
) -> obspy.Stream:
    """
    Get bulk waveforms from waveform client.

    Either 1) call waveform_client's get_waveforms_bulk method (if it exists)
    or 2) iterate the bulk_arg and call get_waveforms for each.

    Parameters
    ----------
    waveform_client
        Any object with a ``get_waveforms`` method.
    bulk
        A list of tuples containing:
            (network, station, location, channel, starttime, endtime)
    """
    if hasattr(waveform_client, "get_waveforms_bulk"):
        return waveform_client.get_waveforms_bulk(bulk)
    # iterate each bulk, collect streams and return
    out = obspy.Stream()
    for bulk_arg in bulk:
        out += waveform_client.get_waveforms(*bulk_arg)
    return out


def _create_stats_df(df, strip_extra=True) -> pd.DataFrame:
    """
    Create a stats dataframe from a list of trace stats objects.

    Parameters
    ----------
    stats_list
        A list of stats objects.
    strip_extra
        If True, strip out columns called "processing" and "response" if
        found.
    """
    out = df.pipe(cast_dtypes, WAVEFRAME_STATS_DTYPES).pipe(
        order_columns, list(WAVEFRAME_STATS_DTYPES)
    )
    # strip extra columns that can have complex object types
    if strip_extra:
        to_drop = ["processing", "response"]
        drop = list(set(to_drop) & set(df.columns))
        out = out.drop(columns=drop)
    return out


def _df_from_stats_waveforms(stats, waveforms):
    """ Get the waveframe df from stats and waveform_client. """

    def _get_data_df(arrays):
        """ A fast way to convert a list of np.ndarrays to a single ndarray. """
        # Surprisingly this is much (5x) times faster than just passing arrays
        # to the DataFrame constructor.
        max_len = max([len(x) if x.shape else 0 for x in arrays])
        out = np.full([len(arrays), max_len], np.NAN)
        for num, array in enumerate(arrays):
            if not array.shape:
                continue
            out[num, : len(array)] = array
        # out has an empty dim just use NaN
        if out.shape[-1] == 0:
            return pd.DataFrame([np.NaN])
        return pd.DataFrame(out)

    def _get_data_and_stats(waveforms, bulk):
        """ Using a waveform client return an array of data and stats. """
        client = get_waveform_client(waveforms)
        # Get a stream of waveforms.
        if not isinstance(waveforms, obspy.Stream):
            waveforms = _get_waveforms_bulk(client, bulk)
        # There isn't guaranteed to be a trace for each bulk arg, so use
        # stream_bulk_split to make it so.
        st_list = stream_bulk_split(waveforms, bulk, fill_value=np.NaN)
        # make sure the data are merged together with a sensible fill value
        arrays, stats = [], []
        for st, b in zip(st_list, bulk):
            assert len(st) in {0, 1}, "st should either be empty or len 1"
            if not len(st):  # empty data still needs an entry and stats from bulk
                arrays.append(np.array(np.NaN))
                statskwargs = {i: v for i, v in zip(BULK_WAVEFORM_COLUMNS, b)}
                stats.append(statskwargs)
                continue
            arrays.append(st[0].data)
            stats.append(dict(st[0].stats))

        return arrays, stats

    # validate stats dataframe and extract bulk parameters
    bulk = get_waveforms_bulk_args(stats)
    # get arrays and stats list
    data_list, stats_list = _get_data_and_stats(waveforms, bulk)
    # then get dataframes of stats and arrays
    stats = _create_stats_df(pd.DataFrame(stats_list))
    ts_df = _get_data_df(data_list)
    return pd.concat([stats, ts_df], axis=1, keys=["stats", "data"])


def _stats_df_to_stats(df: pd.DataFrame) -> List[dict]:
    """
    Convert a stats dataframe back to a list of stats objects.

    Parameters
    ----------
    df
        A dataframe with the required columns.
    """
    # get inputs to cast_dtypes, apply_funcs, drop, etc.
    dtypes = {"starttime": "utcdatetime", "endtime": "utcdatetime"}
    funcs = {"delta": lambda x: x.astype(int) / 1_000_000_000}
    drop = {"sampling_rate"} & set(df.columns)

    df = (
        df.pipe(cast_dtypes, dtypes)
        .pipe(apply_funcs_to_columns, funcs=funcs)
        .drop(columns=drop)
    )
    return df
