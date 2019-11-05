"""
Generic Utilities for Pandas
"""
import fnmatch
import re
from contextlib import suppress
from functools import lru_cache
from typing import Optional, Sequence, Mapping, Union, Collection, Iterable

import numpy as np
import pandas as pd
from obspy import UTCDateTime as UTC

from obsplus.constants import (
    column_function_map_type,
    NULL_SEED_CODES,
    NSLC,
    SMALLDT64,
    LARGEDT64,
)
from obsplus.utils.time import to_datetime64


def convert_bytestrings(df, columns, inplace=False):
    """
    Convert byte strings columns to strings.

    This removes 'b' and quotation marks from string columns. For some reason
    encode doesn't work on data returned from hdf5, hence this approach is a
    bit hacky.

    Parameters
    ----------
    df
        The input dataframe.
    columns
        The names of the columns to convert to string types
    inplace
        If True, perform operation in place.
    """

    def stringitize(ser):
        return ser.astype(str).str.replace("b", "").str.replace("'", "")

    funcs = {x: stringitize for x in columns}
    return apply_funcs_to_columns(df, funcs=funcs, inplace=inplace)


def apply_funcs_to_columns(
    df: pd.DataFrame, funcs: Optional[column_function_map_type], inplace: bool = False
) -> pd.DataFrame:
    """
    Apply callables to columns.

    Parameters
    ----------
    df
        The input dataframe.
    funcs
        A mapping of {column_name, function_to_apply}.
    inplace
        If True, perform operation in place.

    Returns
    -------
    A new dataframe with the columns replaced with output of the function.
    """
    if df.empty:
        return df
    if funcs is not None:
        df = df.copy() if inplace else df
        for col in set(df.columns) & set(funcs):
            df[col] = funcs[col](df[col])
    return df


def cast_dtypes(
    df: pd.DataFrame, dtype: Optional[Mapping[str, type]] = None, inplace=False
) -> pd.DataFrame:
    """
    Cast data types for columns in dataframe, skip columns that doesn't exist.

    Parameters
    ----------
    df
        Dataframe
    dtype
        A dict of columns and datatypes.
    inplace
        If true perform operation in place.
    """
    df = df.copy() if inplace else df
    dtype = {i: dtype[i] for i in set(dtype) & set(df.columns)}
    return df.astype(dtype)


def order_columns(df: pd.DataFrame, required_columns: Sequence, drop_columns=False):
    """
    Given a dataframe, assert that required columns are in the df, then
    order the columns of df the same as required columns with extra columns
    sorted and attached at the end.

    Parameters
    ----------
    df
        The input dataframe.
    required_columns
        A sequence that contains the column names.
    drop_columns
        If True drop columns not in required_columns.
    Returns
    -------
    pd.DataFrame
    """
    # make sure required columns are there
    column_set = set(df.columns)
    extra_cols = sorted(list(column_set - set(required_columns)))
    if drop_columns:  # dont include extras if drop_columns
        extra_cols = []
    new_cols = list(required_columns) + extra_cols
    # add any extra (blank) columns if needed and sort
    df = df.reindex(columns=new_cols)
    return df


def replace_or_swallow(df: pd.DataFrame, replace: dict) -> pd.DataFrame:
    """
    Replace values in a dataframe with new values.

    Parameters
    ----------
    df
        The dataframe for which the values will be replaced
    replace
        A dict of {old_value: new_values}
    """
    if not replace:
        return df
    with suppress(Exception):
        df = df.replace(replace)
    return df


def get_seed_id_series(df: pd.DataFrame, null_codes=NULL_SEED_CODES) -> pd.Series:
    """
    Create a series of seed_ids from a dataframe with required columns.

    The seed id series contains strings of the form:
        network.station.location.channel

    Any "nullish" values (defined by the parameter null_codes) will be
    replaced with an empty string.

    6Parameters
    ----------
    df
        Any Dataframe that has columns with str dtype named:
            network, station, location, channel
    null_codes
        Codes which should be replaced with a blank string.

    Returns
    -------
    A series of concatenated seed_ids codes.
    """
    assert set(NSLC).issubset(df.columns), f"dataframe must have columns {NSLC}"
    replace_dict = {x: "" for x in null_codes}
    nslc = df[list(NSLC)].astype(str).replace(replace_dict)
    net, sta, loc, chan = [nslc[x] for x in NSLC]
    return net + "." + sta + "." + loc + "." + chan


def filter_index(
    index: pd.DataFrame,
    network: Optional = None,
    station: Optional = None,
    location: Optional = None,
    channel: Optional = None,
    starttime: Optional[Union[UTC, float]] = None,
    endtime: Optional[Union[UTC, float]] = None,
    **kwargs,
) -> np.array:
    """
    Filter a waveform index dataframe based on nslc codes and start/end times.

    Parameters
    ----------
    index
        A dataframe to filter which should have the corresponding columns
        to any non-None parameters used in filter.
    network
        A network code as defined by seed standards.
    station
        A station code as defined by seed standards.
    location
        A location code as defined by seed standards.
    channel
        A channel code as defined by seed standards.
    starttime
        The starttime of interest.
    endtime
        The endtime of interest.

    Additional kwargs are used as filters.

    Returns
    -------
    A numpy array of boolean values indicating if each row met the filter
    requirements.
    """
    # handle non-starttime/endtime queries
    query = dict(network=network, station=station, location=location, channel=channel)
    kwargs.update({i: v for i, v in query.items() if v is not None})
    out = filter_df(index, **kwargs)
    # handle starttime/endtime queries if needed
    if starttime is not None or endtime is not None:
        time_out = _filter_starttime_endtime(index, starttime, endtime)
        out = np.logical_and(out, time_out)
    return out


def filter_df(df: pd.DataFrame, **kwargs) -> np.array:
    """
    Determine if each row of the index meets some filter requirements.

    Parameters
    ----------
    df
        The input dataframe.
    kwargs
        Any condition to check against columns of df. Can be a single value
        or a collection of values (to check isin on columns). Str arguments
        can also use unix style matching.

    Returns
    -------
    A boolean array of the same len as df indicating if each row meets the
    requirements.
    """
    # ensure the specified kwarg keys have corresponding columns
    if not set(kwargs).issubset(df.columns):
        msg = f"columns: {set(kwargs) - set(df.columns)} are not found in df"
        raise ValueError(msg)
    # divide queries into flat parameters and collections
    flat_query = {
        k: v
        for k, v in kwargs.items()
        if isinstance(v, str) or not isinstance(v, Collection)
    }
    sequence_query = {
        k: v for k, v in kwargs.items() if k not in flat_query and v is not None
    }
    # get a blank index of True for filters
    bool_index = np.ones(len(df), dtype=bool)
    # filter on non-collection queries
    for key, val in flat_query.items():
        if isinstance(val, str):
            regex = get_regex(val)
            new = df[key].str.match(regex).values
            bool_index = np.logical_and(bool_index, new)
        else:
            new = (df[key] == val).values
            bool_index = np.logical_and(bool_index, new)
    # filter on collection queries using isin
    for key, val in sequence_query.items():
        bool_index = np.logical_and(bool_index, df[key].isin(val))

    return bool_index


def _filter_starttime_endtime(df, starttime=None, endtime=None):
    """ Filter dataframe on starttime and endtime. """
    bool_index = np.ones(len(df), dtype=bool)
    t1 = to_datetime64(starttime) if starttime is not None else SMALLDT64
    t2 = to_datetime64(endtime) if endtime is not None else LARGEDT64
    # get time columns
    start_col = getattr(df, "starttime", getattr(df, "start_date", None))
    end_col = getattr(df, "endtime", getattr(df, "end_date", None))
    in_time = ~((end_col < t1) | (start_col > t2))
    return np.logical_and(bool_index, in_time.values)


@lru_cache(maxsize=2500)
def get_regex(seed_str):
    """ Compile, and cache regex for str queries. """
    return fnmatch.translate(seed_str)  # translate to re


def _column_contains(ser: pd.Series, str_sequence: Iterable[str]) -> pd.Series:
    """ Test if a str series contains any values in a sequence """
    safe_matches = {re.escape(x) for x in str_sequence}
    return ser.str.contains("|".join(safe_matches)).values
