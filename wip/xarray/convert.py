"""
Conversions functions to and from Xarray objects
"""
import functools
import warnings
from collections.__init__ import defaultdict
from typing import Union, Dict, Any, Optional, Collection

import numpy as np
import obspy
import pandas as pd
import xarray as xr
from obspy import Trace, Stream

from obsplus.events.pd import (
    _default_cat_to_df as events_to_df,
    _default_pick_to_df as picks_to_df,
)
from obsplus.constants import wave_type, stream_proc_type, TIME_PRECISION, DIMS
from obsplus.waveforms.utils import trim_event_stream
from obsplus.waveforms.xarray import ops_method


def obspy_to_array_dict(
    waveform: Union[wave_type, Dict[Any, wave_type]],
    trim_stream: bool = True,
    stream_processor: Optional[stream_proc_type] = None,
) -> Dict[int, xr.DataArray]:
    """
    Create a dict of DataArrays from traces with several sampling rates.

    Loads waveforms that potentially have different sampling_rates into a dict
    of DataArrays where the key is the sampling rate and the values are the
    DataArrays.

    Parameters
    ----------
    waveform
        Waveform array (either a Stream or dict/sequence of DataArrays)
        that contain the waveform data
    trim_stream
        If True, make sure all streams
    stream_processor
        A callable for pre-processing the data

    Returns
    -------
    Dict[int, xr.DataArray]
        The dictionary of data arrays
    """
    out = {}
    if isinstance(waveform, (Trace, Stream)):  # a waveforms or trace was passed
        out.update(_split_by_sampling_rate(waveform))
    elif isinstance(waveform, dict):  # a dict of waveforms/dict was passed
        out.update(_iter_events_assemble(waveform))
    func = functools.partial(
        obspy_to_array, trim_stream=trim_stream, stream_processor=stream_processor
    )
    return {item: func(val) for item, val in out.items()}


def _iter_events_assemble(waveform: Dict[Any, Union[Stream, Trace]]):
    """given a dict of the form {event_id: waveforms}, create a new dict of
    the structure {sampling_rage: {event_id: waveforms}}"""
    out = defaultdict(lambda: defaultdict(obspy.Stream))
    for item, value in waveform.items():
        if isinstance(value, Trace):
            value = Stream(traces=[value])
        for tr in value:
            sr = int(np.round(tr.stats.sampling_rate))
            out[sr][item] += tr
    return dict(out)


def _split_by_sampling_rate(
    st: Union[obspy.Stream, obspy.Trace]
) -> Dict[int, obspy.Stream]:
    """given a waveforms, split the waveforms into dicts with unique sampling
    rates"""
    if isinstance(st, obspy.Trace):  # convert to waveforms if trace passed
        st = obspy.Stream(traces=[st])
    # iterate and separate sampling_rates
    out = defaultdict(obspy.Stream)
    for tr in st:
        sr = int(np.round(tr.stats.sampling_rate))
        out[sr] += tr
    return dict(out)


def obspy_to_array(
    waveform: Union[wave_type, Dict[Any, wave_type], Collection[wave_type]],
    trim_stream: bool = True,
    stream_processor: Optional[stream_proc_type] = None,
) -> xr.DataArray:
    """
    Convert a trace, stream, or dict of traces or streams to a data array.

    Parameters
    ----------
    waveform
        The waveform data, can be a waveforms, trace, list of streams/traces
        or a dict of the form {str: Union[waveforms, trace]}.
    trim_stream
        If True trim each waveforms to only include times when all channels have
        contiguous data.
    stream_processor
        A callable that takes a single argument (a stream) and returns a
        stream.
    """
    # if waveforms is a DataArray already just return it
    if isinstance(waveform, xr.DataArray):
        assert set(waveform.dims).issuperset(DIMS)
        return waveform
    # handle converting waveforms
    if isinstance(waveform, Trace):  # if trace convert to waveforms
        waveform = Stream(traces=[waveform])
    if isinstance(waveform, Stream):  # if waveforms convert to dict
        waveform = {0: waveform}
        assert len(waveform), "No data in streams!"
    if not isinstance(waveform, dict):  # hope for collection at this point
        waveform = {x: st for x, st in enumerate(waveform)}
    # at this point waveforms should be a dict
    assert isinstance(waveform, dict), "input not correctly handled"
    # loop and get a list of waveforms data arrays
    out, stats, wf_dict = {}, {}, dict(waveform)
    for item, st in waveform.items():
        if isinstance(st, Trace):
            st = Stream(traces=[st])
            wf_dict[item] = st
        if not len(st):
            continue
        ar = _stream_to_data_array(
            st, trim_stream=trim_stream, stream_processor=stream_processor
        )
        ar.name = item
        out[item] = ar
        stats[item] = ar.attrs["stats"]
        sampling_rate = st[0].stats.sampling_rate
    # trim times to equal lengths (avoids trim off by one errors)
    ds = xr.merge(list(out.values()), join="outer", compat="no_conflicts")
    dar = ds.to_array(dim="stream_id")
    if trim_stream:  # ensure uniform lengths
        dar = _trim_data_array(dar, out)
    # set starttimes, sampling rate and stats
    dar.coords["starttime"] = _get_starttime_df(wf_dict)
    dar.attrs["sampling_rate"] = sampling_rate
    dar.attrs["stats"] = stats
    return dar


def _trim_data_array(dar, array_dict, required_len=0.95):
    """ trim data array to remove short channels and such """
    # determine if there are any channels that are not all NaN or all not
    isvalid = ~dar.isnull()
    all_nan = (~isvalid).all(dim="time")
    no_nan = isvalid.all(dim="time")
    good = all_nan | no_nan
    if not good.all():
        time_mean = isvalid.mean(dim="time")
        st_ids = isvalid.stream_id[time_mean.max(dim="seed_id") > required_len]
        se_ids = isvalid.seed_id[time_mean.max(dim="stream_id") > required_len]
        bad_st_ids = set(dar.stream_id.values) - set(st_ids.values)
        bad_se_ids = set(dar.seed_id.values) - set(se_ids.values)
        if bad_se_ids or bad_st_ids:
            msg = f"removing seed_id {bad_se_ids} and stream_ids {bad_st_ids}"
            warnings.warn(msg)
            dar = dar.loc[st_ids, se_ids, :]
        # pop bad stream_ids out of array_dict
        for bad_st_id in bad_st_ids:
            array_dict.pop(bad_st_id, None)
    # determine if a few samples from the end should be trimmed
    min_lin = min([len(x.time) for x in array_dict.values()])
    if min_lin != len(dar.time):
        dar = dar[:, :, :min_lin]
    return dar


def _stream_to_data_array(
    stream: Stream,
    trim_stream: bool = True,
    stream_processor: Optional[stream_proc_type] = None,
) -> xr.DataArray:
    """
    Convert the waveforms to a data array

    Parameters
    ----------
    stream: obspy.Stream
        The input waveforms to convert
    trim_stream : bool
        If True trim the waveforms to only times when data are available for
        all traces
    stream_processor
        A function to apply to the waveforms before converting to xarray objects

    Returns
    ---------
    xr.DataArray:
        data array containing waveforms info
    """
    # prepare waveforms
    stream.sort()  # always sort for consistent order
    if stream_processor is not None:
        stream = stream_processor(stream)
        assert isinstance(stream, obspy.Stream), "function must return waveforms"
    if trim_stream:
        stream = trim_event_stream(stream)
    sampling_rates = {int(tr.stats.sampling_rate) for tr in stream}
    assert len(sampling_rates) == 1, "channels must have same sampling_rate"
    out, stats = [], {}
    for num, tr in enumerate(stream):
        seedid = tr.id
        ar = _trace_to_datarray(tr)
        ar.name = seedid
        out.append(ar)
        stats[seedid] = ar.attrs["stats"]
    ds = xr.merge(out, join="outer", compat="no_conflicts")
    ar = ds.to_array(dim="seed_id")
    ar.attrs["stats"] = stats
    return ar


def _trace_to_datarray(trace: obspy.Trace) -> xr.DataArray:
    """ convert an obspy trace to a data array oject """
    sr = np.round(trace.stats.sampling_rate, TIME_PRECISION)
    time_stamps = np.arange(0, len(trace.data)) * (1.0 / sr)
    coords = {"time": np.round(time_stamps, TIME_PRECISION)}
    dar = xr.DataArray(trace.data, dims="time", coords=coords)
    dar.attrs["stats"] = trace.stats
    return dar


def _get_starttime_df(waveform):
    """return a df of starttimes with stream_id as coulmns and seed_id as
    index"""
    start_time = {
        (stream_id, tr.id): tr.stats.starttime.timestamp
        for stream_id, st in waveform.items()
        for tr in st
    }
    _time_df = pd.DataFrame(start_time, index=[0]).T.reset_index()

    time_df = _time_df.pivot(values=0, columns="level_0", index="level_1")
    time_df.columns.name = "stream_id"
    time_df.index.name = "seed_id"
    return time_df


@ops_method("to_stream")
def array_to_obspy(dar: xr.DataArray) -> Dict[str, obspy.Stream]:
    """
    Convert a data array back to a dict of obspy streams.

    Parameters
    ----------
    dar
        An xarray DataArray containing waveform information
    """
    out = {}
    assert set(DIMS).issubset(dar.coords)
    assert dar.coords["time"].min() == 0.0, "time must start at 0, use rest_time"
    for stream_id in dar.stream_id.values:
        out[stream_id] = _array_to_stream(dar.sel(stream_id=stream_id), stream_id)
    return out


def _array_to_stream(dar, sid):
    """ convert the data array to a waveforms """
    assert {"time", "seed_id"}.issubset(dar.dims)
    traces = []
    stats = dar.attrs["stats"]
    for sub_dar in dar:
        seed_id = str(sub_dar.seed_id.values)
        try:  # if this stream_id doesnt have this channel
            stat = stats[sid][seed_id]
        except KeyError:
            continue
        try:  # update starttimes using starrtime coord
            starttime = float(sub_dar.starttime.values)
        except (AttributeError, ValueError):
            pass
        else:
            stat["starttime"] = obspy.UTCDateTime(starttime)
        tr = obspy.Trace(data=sub_dar.values)
        tr.stats = stat
        traces.append(tr)
    return Stream(traces=traces)


@ops_method("attach_events")
def attach_events(
    dar: xr.DataArray,
    catalog: Optional[obspy.Catalog] = None,
    event_df: Optional[pd.DataFrame] = None,
    pick_df: Optional[pd.DataFrame] = None,
) -> xr.DataArray:
    """
    Attach A events to channels found in a waveform data array.

    Parameters
    ----------
    dar
        Waveform Data Array
    catalog
        obspy events, optional
    event_df
        dataframe containing event info, eg from :func: `detex.read_events`
    pick_df
        dataframe containing pick info, eg from :func: `
    Returns
    -------
    xr.DataArray
        The standard detex waveform array
    """
    # only a events or event_df/pick_df combo should be defined
    assert (catalog is None) ^ (event_df is None and pick_df is None)
    if catalog is not None:  # get everything in df format
        event_df = events_to_df(catalog)
        pick_df = picks_to_df(catalog)
        dar.attrs["events"] = catalog  # attach ref of events to dar
    # attach events and picks
    _attach_events(dar, event_df)
    _attach_picks(dar, pick_df)
    return dar


def _attach_events(dar, event_df):
    """ attach the interesting info from events onto dataframe """
    # convert stream_ids to pandas, join in origin_times
    df_stream = dar.stream_id.to_pandas().to_frame(name="event_id")
    df_merged = df_stream.merge(
        event_df, how="left", left_on="event_id", right_on="event_id"
    )
    df = df_merged.set_index("event_id")["time"]
    df.name = "stream_id"  # rename event_id to waveforms id for coord alignment
    # set origin_time coords
    dar.coords["origin_time"] = ("stream_id", df)
    return dar


def _attach_picks(dar, pick_df: pd.DataFrame):
    """ join the pick_df to the stream_df """
    phases = ["P", "S"]  # these will be made into coords on dar
    # add seed_id to pick_df
    df = pick_df.copy()
    df["seed_id"] = (
        df["network"] + "." + df["station"] + "." + df["location"] + "." + df["channel"]
    )
    # ensure time contains time stamps
    df["time"] = df["time"].apply(lambda x: obspy.UTCDateTime(x).timestamp)
    # sort and drop duplicates, keeping only the first value
    # this will ensure only the first of each phase for each event/seed is kept
    sort_vals = ["event_id", "seed_id", "phase_hint", "time"]
    df.sort_values(sort_vals, inplace=True)
    df.drop_duplicates(sort_vals[:-1], inplace=True)
    df.reset_index(inplace=True, drop=True)
    # pivot data and assign coords
    for phase in phases:
        df = df[df.phase_hint == phase]
        pivot = df.pivot(index="event_id", columns="seed_id", values="time")
        df_coords = pd.DataFrame(
            index=dar.coords["stream_id"].values,
            columns=dar.coords["seed_id"].values,
            dtype=float,
        )
        df_coords.update(pivot)  # ensures extra index/columns are chopped
        name = phase.lower() + "_time"
        dar.coords[name] = (("stream_id", "seed_id"), df_coords)
