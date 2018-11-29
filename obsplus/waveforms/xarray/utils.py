"""
Utilities for working with xarray data structures.
"""
import copy
import fnmatch

import functools
from functools import partial
from typing import Callable, Union, Optional

import numpy as np
import pandas as pd

import xarray as xr

from obsplus.constants import DIMS, AGG_LEVEL_MAP, TIME_PRECISION, NSLC, xr_type
from obsplus.waveforms.xarray import ops_method


def keep_attrs(func: Callable[..., xr_type]):
    """
    decorator to copy attrs from one xr object (first input) to output
    """

    @functools.wraps(func)
    def new_func(xr_object, *args, **kwargs):
        assert isinstance(xr_object, (xr.DataArray, xr.Dataset))
        attrs = xr_object.attrs
        out = func(xr_object, *args, **kwargs)
        if isinstance(out, (xr.Dataset, xr.DataArray)):
            attrs.update(out.attrs)
            out.attrs = attrs
        return out

    return new_func


@ops_method("trim")
def trim_waveform_array(
    dar: xr.DataArray,
    trim: Union[str, xr.DataArray, float],
    is_timestamp: bool = True,
    remove_nan: bool = True,
    aggregate_by: Optional[str] = None,
    aggregate_func: Callable = np.nanmean,
):
    """
    Trim or extend the waveform array either independently for each waveform
    or  holding relative times between any waveforms constant.

    Can use the name of an existing coordinate (eg 'p_times', 'origin_time'),
    a data array that shares some dimensions with the dar, or number which
    can refer to a time stamp or a time relative to the start of the array.

    Parameters
    ----------
    dar
        The Data Array with waveform vectors
    trim
        The name of a coordinate (all float values) for trimming, a single
        float, or another data array (which shares some dims with dar)
        that contains either timestamps or relative trim times.
    is_timestamp
        If True, the values given by trim are timestamps
        (ie seconds from 1970). If False they are relative to the traces
        start time.
    remove_nan
        If True remove the NaN values from the end of the arrays
    aggregate_by
        If not None, must be the name of a coord in dar (eg "group") or a value
        supported by the function obsplus.waveforms.xarray.ggregate.
    aggregate_func
        Any function that takes a numpy array and returns a single value.
        Since NaNs are possible it is wise to use functions that account
        for them.
    Returns
    -------
    xr.DataArray
        The trimmed data array
    """
    # ensure the expected dimensions are there and the trim coord exists
    assert set(DIMS).issubset(dar.coords), f"{DIMS} must be in coords"
    assert "starttime" in dar.coords, "data array must have a starttime coord"
    trim_array = _get_trim_data_array(dar, trim, is_timestamp)
    # temporarily attach trim_array to dar
    dar.coords["_trim"] = trim_array
    # apply aggregations if needed on trim times
    if aggregate_by:
        if aggregate_by in AGG_LEVEL_MAP:
            dar, _ = _add_level(dar, aggregate_by)
        func = partial(_overwrite_group_values, func=aggregate_func)
        group = dar.coords["_trim"].groupby(aggregate_by)
        dar.coords["_trim"] = group.apply(func)
    # stack and group by dims in the _trim coord
    stacked = dar.stack(z=dar.coords["_trim"].dims)
    gr = stacked.groupby("z")
    # apply function to each group and unstack
    out_stacked = gr.apply(_trim_array)
    try:  # fails to unstack when stacked on one dimension
        out = out_stacked.unstack("z")
    except ValueError:  # see xarray issue # 1431
        out = out_stacked.stack(**{dar.coords["_trim"].dims[0]: ("z",)})
    del dar.coords["_trim"]  # drop temporary coord from original array
    return _prepare_trim_output(dar, out, remove_nan)


def _prepare_trim_output(dar, out, remove_nan):
    """ prepare the output of trim function by adjusting starttimes, deleting
    temporary coord, and removing nan output """
    # TODO remove this when xarray issue # 1428 gets resolved
    if (dar.starttime == out.starttime).all():  # starttimes didnt update
        fill_values = out.coords["_trim"].fillna(0.0)
        out.coords["starttime"] = out.coords["starttime"] + fill_values
    del out.coords["_trim"]  # remove temporary coord
    out = out.transpose(*DIMS)
    if remove_nan:
        out = out.dropna("time")
    return out


def _overwrite_group_values(dar: xr.DataArray, func: Callable):
    """ apply function on a data array, set all values to func output """
    aggregate = func(dar)
    dar.values = np.ones_like(dar.values) * aggregate
    return dar


def _get_trim_data_array(dar, trim, is_timestamp):
    """ return a data array that shares some dims with dar for trimming.
    Values in data array will be referenced from the start of each waveform
    rather than a timestamp"""
    # figure out what trim is turn it into a data array
    if isinstance(trim, str):  # trim is a coord in dar
        assert trim in dar.coords, f"{trim} must be in coords"
        trim = dar.coords[trim]
    elif isinstance(trim, (float, int)):  # make homogeneous data array
        trim = xr.ones_like(dar.starttime) * trim
    elif isinstance(trim, xr.DataArray):  # must contain common dims
        assert set(trim.dims).issubset(dar.dims)
    if is_timestamp:  # get absolute time into relative time
        trim = trim - dar.starttime
    return trim


def _trim_array(dar: xr.DataArray):  # , trim_coord: str,
    # reference_starttime: bool, conglomerate_func=None):
    """ apply the trim function to array """
    trim_value = -float(dar.coords["_trim"])  # reverse sign for pad_zero func
    if pd.isnull(trim_value) or not trim_value:  # skip nan and 0s (no effect)
        out = dar
    else:
        out = pad_time(dar, time_before=trim_value, start_at_zero=True)
    # eliminate roundoff error on time vector
    out.coords["time"].values = np.around(out.time.values, TIME_PRECISION)
    return out


def get_nslc_df(dar: xr.DataArray):
    """ return a dataframe with network station location channel
    from seed_id on a detex data array"""
    ser = dar.seed_id.to_pandas() if isinstance(dar, xr.DataArray) else dar
    ser_ind = pd.Series(np.array(ser.index), index=ser.index)
    df = ser_ind.str.split(".", expand=True)
    df.columns = NSLC
    return df


@ops_method("stack_seed")
def stack_seed(dar: xr.DataArray, level) -> xr.DataArray:
    """
    Stack the DataArray on a defined level.

    Parameters
    ----------
    dar
        An obsplus data array
    level
        The seed-level (network, station, location or channel).
    """
    dar = dar.copy()  # changes some things in place, make a copy
    # This is super ugly, and much harder than it should be.
    # get dataframe of seed levels
    assert level in NSLC
    df = get_nslc_df(dar)
    # create multi-index and swap out old seed_id for multi-level
    ind = pd.MultiIndex.from_arrays(
        (df[level].values, df.index.values), names=(level, "sid")
    )
    dar.seed_id.values = ind
    # unstack, drop nans, then stack ids together
    out = dar.unstack("seed_id").rename({"sid": "seed_id"})
    out = out.stack(ids=("seed_id", "stream_id"))
    return out.dropna(dim="ids", how="all").transpose("ids", level, "time")


@ops_method("unstack_seed")
def unstack_seed(dar: xr.DataArray) -> xr.DataArray:
    """
    Unstack the DataArray stacked on a seed level.

    The DataArray should have been created with the functon:
    `~:func:obsplus.waveforms.xarray.utils.stack_seed`

    Parameters
    ----------
    dar
        An obsplus data array
    """
    # This is also super ugly, and much harder than it should be.
    # find level that should be squished
    level = set(NSLC) & set(dar.dims)
    if len(level) != 1:
        msg = "could not determine how to unstack data array based on seed level"
        raise ValueError(msg)
    level = list(level)[0]
    # stack seed_id with level, remove multi-index
    stack = ("seed_id", level)
    dar1 = dar.unstack().stack(sid=stack).dropna(dim="sid", how="all")
    dar1.sid.values = [x[0] for x in dar1.sid.values]
    return dar1.rename({"sid": "seed_id"}).transpose(*DIMS)


@ops_method("sel_sid")
def sel_sid(dar: xr.DataArray, seed_id):
    """
    Slice a detex Data Array based on a seed_id.

    Seed id normally has the form: network.station.locaton.channel and
    allows for unix style search strings. Lower level ids can be excluded
    for implicit wildcards.

    Parameters
    ----------
    dar
        Data array, generated by :function: ~detex.utils.waveform2array
    seed_id
        The str specifying the seed id. Allows truncations and wildcards.

    Notes
    --------
    the following are all valid values for seed_id:
        BW - select all data with BW as network
        BW.RJOB - select all data from station RJOB on network BW
        BW.RJOB..EHZ - select only data from this channel
        BW.R* - select all station in BW that start with an R
        BW.???B - select all station in BW with 4 chars that end with B

    Returns
    -------
    xr.DataArray
        A DataArray containing the desired seed_ids
    """
    assert "seed_id" in dar.dims, "data array must have seed_id in dims"
    # convert seed_id to pandas objects for matching
    ser = dar.seed_id.to_pandas()
    bool_ms = seed_sel_from_series(ser, seed_id)
    return dar.sel(seed_id=ser.isin(list(bool_ms)).values)


def seed_sel_from_series(ser: Union[pd.Series, xr_type], seed_id):
    """
    Given a series with keys as seed ids, return a view of of the series
    where the seed_ids match seed_str.

    Parameters
    ----------
    ser
        pd.Series
    seed_str
        String for matching seeds

    Returns
    -------
    A view of the series

    """
    ms = pd.Series({item: val for item, val in zip(NSLC, seed_id.split("."))})
    df = get_nslc_df(ser)
    # iter the match series and filter data array
    bool_ms = df.network.astype(bool)
    for ind, val in ms.items():
        # get a bool array if values match regex
        if ms[ind]:  # for non empty str
            new = df[ind].str.contains(fnmatch.translate(ms[ind]))
        else:  # for empty strings explicit equal must be done
            new = df[ind] == ms[ind]
        bool_ms = bool_ms & new
    return ser.loc[bool_ms]


@ops_method("iter_seed")
def iter_seed(dar: xr.DataArray, level: str):
    """
    Iterate over a data array by seed_id.

    Parameters
    ----------
    dar
        A DataArray with waveform information
    level
        The level over which chunks should be returned. Valid options are:
            'network', 'station', 'location', 'channel'

    Yields
    -------
    xr.DataArray
        Slices of dar divided along the seed_id dim according to the level
        argument
    """
    assert level in NSLC, f"level argument not valid must be in {NSLC}"
    assert "seed_id" in dar.dims, "data array must have seed_id in dims"
    df = get_nslc_df(dar)
    uniques = df[level].unique()
    for unique in uniques:
        bool_ser = df[level] == unique
        yield dar.sel(seed_id=bool_ser.values)


@ops_method("pad")
@keep_attrs
def pad_time(
    ar1: xr.DataArray,
    time_after: Optional[float] = None,
    time_before: Optional[float] = None,
    total_time: Optional[float] = None,
    fill_value: float = 0.0,
    start_at_zero: bool = False,
) -> xr.DataArray:
    """
    Pad a DataArray along the time dimension

    Parameters
    ----------
    ar1 : xr.DataArray
        The Data Array which will have zeros appended in the time dimension.
    time_after : float, optional
        The total time after the current end of the trace the array should
        be padded.
    time_before : float, optional
        The time (in seconds) to pad the array before the current start
        time. The starttime coord will be updated if not None. Can be
        negative in order to trim the array.
    total_time : float, optional
        The total time the array should be. If shorter than current length
        the array will be trimmed. Note: if total_time is not None then
        time_after and time before must be None.
    fill_value : float
        The value with which to pad the array.
    start_at_zero : bool
        If True, set the start time to zero regardless of the time before
        parameter. If the coord starttime is in the array it will be adjusted
        to reflect the change.
    Returns
    -------
    xr.DataArray
        The new data array padded with zeros
    """
    # perform parameter checks
    assert "time" in ar1.dims
    assert time_before or time_after or total_time
    assert bool(time_before or time_after) != bool(total_time)
    # get sampling rate from attrs, or infer (assuming evenly spaced)
    sr = 1.0 / ar1.attrs.get("sampling_rate") or ar1.time[1] - ar1.time[0]
    time = ar1.coords["time"].values
    if total_time:  # re-sample to total time
        t1 = np.min(time)
        t2 = t1 + total_time
    else:  # get nearest time divisible by sample rate
        t1 = time[0] - np.round((time_before or 0) / sr) * sr
        t2 = np.round((time_after or 0) / sr) * sr + time[-1]
    new_time = np.arange(t1, t2 + sr, sr)
    kwargs = dict(time=new_time, method="nearest", tolerance=TIME_PRECISION)
    out = ar1.reindex(**kwargs).fillna(fill_value)
    out.values = np.array(out.values)  # bottleneck segfaults without this
    assert np.any(~pd.isnull(out.values)), "re-sampling failed"
    if start_at_zero:
        out = reset_time(out, inplace=True)
    return out


def reset_time(dar: xr.DataArray, inplace=False) -> xr.DataArray:
    """
    Reset the time coord of a DataArray to start at zero.

    Also update starttime coord.

    Parameters
    ----------
    dar
        The data array with a time coord
    inplace
        If True modify the data array in place

    Returns
    -------
    xr.DataArray
        The data array with an adjusted time vector
    """
    assert "time" in dar.coords, "dar must have a time coordinate"
    dar = dar if inplace else copy.deepcopy(dar)  # copy if not inplace
    min_time = float(dar.time.min())
    dar.time.values -= min_time
    if "starttime" in dar.coords:
        dar.coords["starttime"] += min_time
    # eliminate roundoff error on time vector
    dar.coords["time"].values = np.around(dar.time.values, TIME_PRECISION)
    return dar


def _add_level(dar, level, dim="seed_id", coord=None):
    """ given a data array, add the level the aggregation is to occur on
     to the coord """
    new_coord = None
    assert level in AGG_LEVEL_MAP
    if level == "all":  # set all the seed_ids to 1
        dar.coords[level] = (dim, np.ones_like(dar.coords[dim]))
        return dar, new_coord
    ser = getattr(dar, dim).to_dataframe()[dim]
    new = ser.apply(lambda x: ".".join(x.split(".")[: AGG_LEVEL_MAP[level]]))
    dar.coords[level] = (dim, new.values)
    if coord:  # get the new group that will be used as a coord later on
        new_coord = _adjust_coord(dar, coord, new, level, dim)
    return dar, new_coord


def _adjust_coord(dar, coord, coord_map, level, dim):
    """ create new dataframe that will be joined back to aggregated data
    array """
    # TODO this is ugly and probably not efficient, clean up
    assert coord in dar.coords, f"{coord} is not found in coord of {dar}"
    # convert to dataframe and replace any full seed str with substr
    oval = dar[coord].to_pandas()
    if isinstance(oval, pd.Series):
        oval = pd.DataFrame(oval).T
    for item, value in coord_map.items():
        for col in oval.columns:
            oval[col] = oval[col].str.replace(item, value)
    # reshape into new shape of aggregated dataarray
    if not oval.columns.name == dim:
        oval = oval.T
    oval.columns = dar.coords[level]
    oval = oval.T.drop_duplicates().T
    if len(oval) == 1:
        return oval.loc[0]
    return oval
