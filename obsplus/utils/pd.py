"""
Generic Utilities for Pandas
"""
import fnmatch
import re
from contextlib import suppress
from functools import lru_cache, reduce
from typing import Any, Optional, Sequence, Mapping, Collection, Iterable, Union

import numpy as np
import obspy
import pandas as pd
from pandas.api.types import is_string_dtype

from obsplus.constants import (
    column_function_map_type,
    NULL_SEED_CODES,
    NSLC,
    SMALLDT64,
    LARGEDT64,
    utc_time_type,
    BULK_WAVEFORM_COLUMNS,
    bulk_waveform_arg_type,
)
from obsplus.exceptions import DataFrameContentError
from obsplus.utils.time import to_datetime64, to_timedelta64, to_utc


def _int_column_to_str(ser, width=2, fillchar="0"):
    """Convert an int column to a string"""
    # Do nothing if the column is already a string
    if is_string_dtype(ser):
        return ser
    return ser.astype(str).str.pad(width=width, fillchar=fillchar)


# maps obsplus datatypes to functions to apply to columns to obtain dtype
OPS_DTYPE_FUNCS = {
    "ops_datetime": to_datetime64,
    "ops_timedelta": to_timedelta64,
    "utcdatetime": to_utc,
    "location_code": _int_column_to_str,
}

# the dtype of the columns
OPS_DTYPES = {
    "ops_datetime": "datetime64",
    "ops_timedelta": "timedelta64",
    "utcdatetime": obspy.UTCDateTime,
    "location_code": str,
}


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
        A mapping of {column_name: function_to_apply}.
    inplace
        If True, perform operation in place.

    Returns
    -------
    A new dataframe with the columns replaced with output of the function.
    """
    if df.empty:
        return df
    if funcs is not None:
        df = df if inplace else df.copy()
        for col in set(df.columns) & set(funcs):
            df[col] = funcs[col](df[col])
    return df


def _time_cols_to_ints(df, columns=None, nat_value=SMALLDT64):
    """
    Converts all datetime columns to ints.

    Previously NaT were convertible to ints, but now they raise a value error.
    We need, therefore, to designate a time that will be used as NaT.
    """
    cols = columns or df.select_dtypes(include=["datetime64"]).columns
    df.loc[:, cols] = df.loc[:, cols].fillna(nat_value).astype(np.int64)
    return df


def _ints_to_time_columns(df, columns=None, nat_value=SMALLDT64):
    """
    Converts ints in columns (or all ints) to datetimes.

    Needs a fill value for NaT.
    """
    dtypes = [int, np.int64]
    cols = columns or df.select_dtypes(include=dtypes).columns
    df.loc[:, cols] = (
        df.loc[:, cols]
        .apply(pd.to_datetime, unit="ns", axis=1)
        .replace(nat_value, np.datetime64("NaT"))
    )
    return df


def cast_dtypes(
    df: pd.DataFrame,
    dtype: Optional[Mapping[str, Union[type, str]]] = None,
    inplace=False,
) -> pd.DataFrame:
    """
    Cast data types for columns in dataframe, skip columns that doesn't exist.

    The following obsplus specific datatypes are supported:
        'ops_datetime' - call :func:`obsplus.utils.time.to_datetime64` on column
        'ops_timedelta` - call :func:`obsplus.utils.time.to_timedelta64` on column

    Note: this is different from pd.astype because it skips columns which
    don't exist.

    Parameters
    ----------
    df
        Dataframe
    dtype
        A dict of columns and datatypes.
    inplace
        If true perform operation in place.
    """
    # get overlapping columns and dtypes
    overlap = set(dtype) & set(df.columns)
    dtype_codes = {i: dtype[i] for i in overlap}
    # if the dataframe is empty and has columns use simple astype
    if df.empty and len(df.columns):
        dtypes = {i: OPS_DTYPES.get(v, v) for i, v in dtype_codes.items()}
        return df.astype(dtypes)
    # else create functions and apply to each column
    funcs = {
        i: OPS_DTYPE_FUNCS.get(v, lambda x, y=v: x.astype(y))
        for i, v in dtype_codes.items()
    }
    return apply_funcs_to_columns(df, funcs=funcs, inplace=inplace)


def order_columns(
    df: pd.DataFrame, required_columns: Sequence, drop_columns=False, fill_missing=True
):
    """
    Order a dataframe's columns and ensure it has required columns.

    Parameters
    ----------
    df
        The input dataframe.
    required_columns
        A sequence that contains the column names.
    drop_columns
        If True drop columns not in required_columns.
    fill_missing
        If True, create missing required columns and fill with nullish values.

    Returns
    -------
    pd.DataFrame
    """
    # make sure required columns are there
    column_set = set(df.columns)
    missing_cols = set(required_columns) - set(df.columns)
    extra_cols = sorted(list(column_set - set(required_columns)), key=lambda x: str(x))
    if drop_columns:  # dont include extras if drop_columns
        extra_cols = []
    # raise a DataFrameContentError if required columns are not there
    if missing_cols and not fill_missing:
        msg = f"dataframe is missing required columns: {missing_cols}"
        raise DataFrameContentError(msg)
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


def join_str_columns(
    df: pd.DataFrame, columns: Sequence[str], join_char: str = "."
) -> pd.Series:
    """
    Join string columns on a dataframe together.

    Parameters
    ----------
    df
        The input dataframe with columns listed in columns parameter.
    columns
        The columns to be joined. Must be part of df.
    join_char
        The string to join the columns together.
    """
    if len(columns) < 2:
        msg = "at least 2 columns are needed to join"
        raise ValueError(msg)
    slist = [df[x].astype(str) for x in columns]
    return reduce(lambda x, y: x + join_char + y, slist[1:], slist[0])


def get_seed_id_series(
    df: pd.DataFrame,
    null_codes: Optional[Any] = NULL_SEED_CODES,
    subset: Optional[Sequence[str]] = None,
) -> pd.Series:
    """
    Create a series of seed_ids from a dataframe with required columns.

    The seed id series contains strings of the form:
        network.station.location.channel

    Any "nullish" values (defined by the parameter null_codes) will be
    replaced with an empty string.

    Parameters
    ----------
    df
        Any Dataframe that has columns with str dtype named:
            network, station, location, channel
    null_codes
        Codes which should be replaced with a blank string.
    subset
        Used to select a subset of the full seed_id. For example,
        ('network', 'station') would return a series of network.station.

    Returns
    -------
    A series of concatenated seed_ids codes.

    Examples
    --------
    >>> import obsplus
    >>> import obspy
    >>> # Get a dataframe with only network station location channel columns
    >>> cat = obspy.read_inventory()
    >>> NSLC = ['network', 'station', 'location', 'channel']
    >>> df = obsplus.stations_to_df(cat)[NSLC]
    >>> out = get_seed_id_series(df)
    >>> # Get a series of network.station
    >>> net_sta = get_seed_id_series(df, subset=('network', 'station'))
    """
    # first ensure subset is in standard NSLC codes
    if subset is not None and not set(subset).issubset(set(NSLC)):
        msg = f"subset must be a subset of {NSLC}, you passed {subset}"
        raise ValueError(msg)
    # get requested columns and check for their existence
    cols = NSLC if subset is None else tuple(subset)
    if not set(cols).issubset(df.columns):
        missing = set(cols) - set(df.columns)
        msg = f"dataframe is missing specified columns: {missing}"
        raise ValueError(msg)
    # replace nullish codes
    replace_dict = {x: "" for x in null_codes}
    nslc = df[list(cols)].astype(str).replace(replace_dict)
    # join string columns and return
    return join_str_columns(nslc, columns=cols, join_char=".")


def filter_index(
    index: pd.DataFrame,
    network: Optional = None,
    station: Optional = None,
    location: Optional = None,
    channel: Optional = None,
    starttime: Optional[utc_time_type] = None,
    endtime: Optional[utc_time_type] = None,
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


def get_waveforms_bulk_args(
    df: pd.DataFrame, time_dtype: str = "utcdatetime"
) -> bulk_waveform_arg_type:
    """
    Get the inputs to a get_waveforms_bulk from a dataframe.

    Parameters
    ----------
    df
        A dataframe with required columns:
            network, station, location, channel, starttime, endtime
    time_dtype
        Dtype to use for the starttime and endtime

    Returns
    -------
    A list of tuples [(network, station, location, channel, starttime, endtime),]
    """

    def rename_startdate_enddate(df):
        """ rename startdate, enddate to starttime endtime """
        col_set = set(df.columns)
        if "startdate" in col_set and "starttime" not in col_set:
            df = df.rename(columns={"startdate": "starttime"})
        if "enddate" in col_set and "endtime" not in col_set:
            df = df.rename(columns={"enddate": "endtime"})
        return df

    def _times_to_utc(df):
        """ Convert time columns to UTCDateTime."""
        df["starttime"] = to_utc(df["starttime"])
        df["endtime"] = to_utc(df["endtime"])
        return df

    def _check_nslc_codes(df):
        """ Ensure there are no wildcards in NSLC columns. """
        for code in NSLC:
            has_qmark = df[code].str.contains("?", regex=False).any()
            has_star = df[code].str.contains("*", regex=False).any()
            if has_qmark or has_star:
                msg = f"columns {NSLC} cannot contain * or ?, column {code} does"
                raise DataFrameContentError(msg)
        return df

    def _check_starttime_endtime(df):
        """ Ensure all starttimes are less than endtimes. """
        # starttimes must be <= endtime
        invalid_time_range = df["starttime"] >= df["endtime"]
        if invalid_time_range.any():
            msg = "all values in starttime must be <= endtime"
            raise DataFrameContentError(msg)
        return df

    def _check_missing_data(df):
        """ There should be no missing data in the required columns."""
        # first check if all required columns exist
        if not set(BULK_WAVEFORM_COLUMNS).issubset(set(df.columns)):
            missing_cols = set(BULK_WAVEFORM_COLUMNS) - set(df.columns)
            msg = f"Dataframe is missing the following columns: {missing_cols}"
            raise DataFrameContentError(msg)
        missing_date = df[list(BULK_WAVEFORM_COLUMNS)].isnull().any()
        no_data_cols = missing_date[missing_date].index
        if not no_data_cols.empty:
            msg = f"dataframe is missing values in columns: {list(no_data_cols)}"
            raise DataFrameContentError(msg)
        return df

    dtypes = {x: str for x in NSLC}
    dtypes.update({"starttime": time_dtype, "endtime": time_dtype})
    order_cols_kwargs = dict(drop_columns=True, fill_missing=False)

    df = (
        rename_startdate_enddate(df)
        .pipe(_check_missing_data)
        .pipe(_times_to_utc)
        .pipe(order_columns, BULK_WAVEFORM_COLUMNS, **order_cols_kwargs)
        .pipe(_check_nslc_codes)
        .pipe(_check_starttime_endtime)
        .pipe(cast_dtypes, dtypes)
    )
    # df = order_columns(df, required_columns=BULK_WAVEFORM_COLUMNS)

    return df[list(BULK_WAVEFORM_COLUMNS)].to_records(index=False).tolist()
