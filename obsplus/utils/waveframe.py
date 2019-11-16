"""
Waveframe specific utilities.
"""
import obspy
import pandas as pd
from typing import Union, List, Tuple

import numpy as np
from obsplus.constants import TIME_COLUMNS
from obsplus.constants import WaveformClient, bulk_waveform_arg_type, wave_type
from obsplus.utils.time import to_utc
from obsplus.utils.pd import apply_funcs_to_columns, order_columns, cast_dtypes
from obsplus.utils.waveforms import get_waveform_client, stream_bulk_split
from obsplus.constants import BULK_WAVEFORM_COLUMNS, WAVEFRAME_STATS_DTYPES


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


def _get_data_and_stats(
    waveforms: Union[wave_type, pd.DataFrame], bulk
) -> Tuple[List[np.ndarray], List[dict]]:
    """ Given a waveform client return a list of data (np arrays) and stats. """
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
            arrays.append(statskwargs)
            continue
        arrays.append(st[0].data)
        stats.append(st[0].stats)

    return arrays, stats


def _create_stats_df(stats_list):

    """
    Augment the stats dataframe with stats from list.
    """
    df = (
        pd.DataFrame(stats_list)
        .pipe(cast_dtypes, WAVEFRAME_STATS_DTYPES)
        .pipe(order_columns, list(WAVEFRAME_STATS_DTYPES))
        .pipe()
    )

    breakpoint()
