"""
Stream utilities
"""
import copy

import warnings
from functools import singledispatch
from pathlib import Path
from typing import Optional, Union, List, Any

import numpy as np
import obspy
import pandas as pd
from obspy import Stream, UTCDateTime

import obsplus
from obsplus.constants import (
    waveform_clientable_type,
    wave_type,
    NSLC,
    trace_sequence,
    NUMPY_FLOAT_TYPES,
    NUMPY_INT_TYPES,
    waveform_request_type,
    WAVEFORM_REQUEST_DTYPES,
)
from obsplus.exceptions import ValidationError
from obsplus.interfaces import WaveformClient
from obsplus.utils.pd import filter_index
from obsplus.utils.pd import get_seed_id_series, cast_dtypes, _column_contains
from obsplus.utils.time import to_utc, to_timedelta64, to_datetime64
from obsplus.utils.misc import ObjectWrapper


# ---------- trim functions


def trim_event_stream(
    stream: Stream,
    merge: Optional[int] = 1,
    copy: bool = True,
    trim_tolerance=None,
    required_len: Optional[float] = 0.95,
):
    """
    Trim the waveforms to a common start time and end time.

    Uses latest start and earliest end for each trace, unless an abnormally
    short trace is found.

    Parameters
    ----------
    stream : obspy.Stream
        The waveforms to trim
    merge : int, optional
        If not None, merge the waveforms with this method before trimming. See
        obspy waveforms docs for merge options
    copy : bool
        Copy the waveforms before altering it.
    trim_tolerance:
        The number of seconds the trimmed starttime or endtime can vary from
        the starttime/endtime in the current traces.
    required_len
        The fraction of the longest len a trace must have to be considered
        good. If any trace is shorter remove it.

    Returns
    -------
    stream
        The merged waveforms
    """
    if copy:
        stream = stream.copy()
    if merge is not None:  # merge
        stream.merge(method=merge)
        stream = stream.split()
    # get a dataframe of start/end/duration and trace
    data = [
        (tr.stats.starttime.timestamp, tr.stats.endtime.timestamp, tr) for tr in stream
    ]
    df = pd.DataFrame(data, columns=["start", "end", "trace"])
    df["duration"] = df.end - df.start
    # get start and end times
    stream = _trim_stream(df, stream, required_len, trim_tolerance)
    # ensure each channel has exactly one trace or merge to create masked
    if not len(stream) == len({tr.id for tr in stream}):
        stream.merge(method=merge)
    return stream


def _trim_stream(df, stream, required_len, trim_tolerance):
    """
    Get the starttimes and endtimes for trimming, raise ValueError
    if the stream is disjointed.
    """
    # check trim tolerance
    if trim_tolerance is not None:
        con1 = (df.start.max() - df.start.min()) > trim_tolerance
        con2 = (df.end.max() - df.start.min()) > trim_tolerance
        if con1 or con2:
            msg = (
                "the following waveforms did not meed the required trim "
                f"tolerance{str(stream)}"
            )
            raise ValueError(msg)
    # check length requirements, pop out any traces that dont meet it
    if required_len is not None:
        req_len = np.round(required_len * df.duration.max(), 2)
        too_short = df.duration <= req_len
        if too_short.any():
            trace_str = "\n".join([str(x) for x in df[too_short].trace])
            msg = f"These traces are not at least {req_len} seconds long:\n"
            warnings.warn(msg + trace_str + "\n removing them", UserWarning)
            stream.traces = list(df[~too_short].trace)
        df = df[~too_short]
    if not len(df):
        return Stream()
    # get trim time, trim, emit warnings
    t1, t2 = to_utc(df.start.max()), to_utc(df.end.min())
    if t2 < t1:
        msg = f"The following waveforms has traces with no overlaps {stream}"
        raise ValueError(msg)
    return stream.trim(starttime=t1, endtime=t2)


def _get_waveform_df(stream: wave_type) -> pd.DataFrame:
    """
    Convert a stream of sequence of traces into a datframe.

    Parameters
    ----------
    stream
        The streams to index

    Notes
    -----
    This is private because it is probably not quite polished enough to include
    in the public API. More thought is needed how to do this properly.
    """
    stats_columns = list(NSLC) + ["starttime", "endtime", "sampling_rate"]
    trace_contents = [{i: tr.stats[i] for i in stats_columns} for tr in stream]
    df = pd.DataFrame(trace_contents, columns=stats_columns)
    # ensure time(y) columns have proper
    df["starttime"] = to_datetime64(df["starttime"])
    df["endtime"] = to_datetime64(df["endtime"])
    df["sampling_period"] = to_timedelta64(1 / df["sampling_rate"])
    df["seed_id"] = get_seed_id_series(df)
    df["trace"] = [ObjectWrapper(tr) for tr in stream]
    return df


def _get_bulk(bulk):
    """Private function to reformat the bulk"""
    # Convert dataframe to bulk input
    if bulk is None:
        return []
    if isinstance(bulk, pd.DataFrame):
        cols = list(NSLC) + ["starttime", "endtime"]
        assert set(cols).issubset(
            bulk.columns
        ), f"A bulk dataframe must have the following columns: {cols}"
        return list(bulk[cols].to_records(index=False))
    return bulk


def stream_bulk_split(
    st: Stream, bulk: List[waveform_request_type], fill_value: Any = None
) -> List[Stream]:
    """
    Split a stream into a list of streams that meet requirements in bulk.

    This is similar to the get_waveforms_bulk methods of waveform_client, but
    rather than merging any overlapping data it is returned in a list of traces.

    Parameters
    ----------
    st
        A stream object
    bulk
        A bulk request. Wildcards not currently supported on str params.
    fill_value
        If not None fill any missing data in time range with this value.

    Returns
    -------
    List of traces, each meeting the corresponding request in bulk.
    """
    # return nothing if empty bulk or stream args
    bulk = _get_bulk(bulk)
    if not bulk or len(st) == 0:
        return []

    # # get dataframe of stream contents
    sdf = _get_waveform_df(st)
    # iterate stream, return output
    out = []
    for barg in bulk:
        assert len(barg) == 6, f"{barg} is not a valid bulk arg, must have len 6"
        need = filter_index(sdf, *barg)
        traces = [tr for tr, bo in zip(st, need) if bo]
        new_st = obspy.Stream(traces)
        t1, t2 = to_utc(barg[-2]), to_utc(barg[-1])
        new = new_st.slice(starttime=t1, endtime=t2)
        # apply fill if needed
        if fill_value is not None:
            new = new.trim(starttime=t1, endtime=t2, fill_value=fill_value, pad=True)
        if new is None or not len(new):
            out.append(obspy.Stream())
            continue
        new = merge_traces(new)
        out.append(new)
    assert len(out) == len(bulk), "output is not the same len as stream list"
    return out


def merge_traces(st: trace_sequence, inplace=False) -> obspy.Stream:
    """
    An efficient function to merge overlapping data for a stream.

    This function is equivalent to calling merge(1) and split() then returning
    the resulting trace. This means only traces that have overlaps or adjacent
    times will be merged, otherwise they will remain separate traces.

    Parameters
    ----------
    st
        The input stream to merge.
    inplace
        If True st is modified in place.

    Returns
    -------
    A stream with merged traces.
    """

    def _make_trace_df(traces: trace_sequence) -> pd.DataFrame:
        """Create a dataframe form a sequence of traces."""
        # create dataframe of traces and stats (use ns for all time values)
        sortby = ["seed_id", "sampling_rate", "starttime", "endtime"]
        index = _get_waveform_df(traces).sort_values(sortby).reset_index(drop=True)
        # create column if trace should be merged into previous trace
        shifted_index = index.shift(1)  # shift all value forward one column
        seed_ids_match = index["seed_id"] == shifted_index["seed_id"]
        index_samp = index["sampling_period"]
        samps_match = index_samp == shifted_index["sampling_period"]
        # not sure why ~ is used rather than flipping comparison, maybe something
        # with NaNs?
        overlap = ~(shifted_index["endtime"] < (index["starttime"] - index_samp))
        index["merge_group"] = (~(seed_ids_match & samps_match & overlap)).cumsum()
        return index

    # checks for early bail out, no merging if one trace or all unique ids
    if len(st) < 2 or len({tr.id for tr in st}) == len(st):
        return st
    if not inplace:
        st = copy.deepcopy(st)
    df = _make_trace_df(st)
    # get a series of properties by group
    group = df.groupby("merge_group")
    t1, t2 = group["starttime"].min(), group["endtime"].max()
    sampling_periods = group["sampling_period"].max()
    gsize = group.size()  # number of traces in each group
    gnum_one = gsize[gsize == 1].index  # groups with 1 trace
    gnum_gt_one = gsize[gsize > 1].index  # group num with > 1 trace
    # use this to avoid pandas groupbys
    merged_traces = [x.data for x in df[df["merge_group"].isin(gnum_one)]["trace"]]
    for gnum in gnum_gt_one:  # any groups w/ more than one trace
        ind = df.merge_group == gnum
        gtraces = [x.data for x in df.trace[ind]]  # unpack traces
        dtype = _get_dtype(gtraces)
        # create list of time, y values, and marker for when values are filled
        sampling_period = sampling_periods[gnum].to_timedelta64()
        start, stop = t1[gnum].to_datetime64(), t2[gnum].to_datetime64()
        t = np.arange(start=start, stop=stop + sampling_period, step=sampling_period)
        y = np.empty(np.shape(t), dtype=dtype)
        has_filled = np.zeros(len(t), dtype=np.int32)
        for tr in gtraces:
            start_ind = np.searchsorted(t, to_datetime64(tr.stats.starttime))
            y[start_ind : start_ind + len(tr.data)] = tr.data
            has_filled[start_ind : start_ind + len(tr.data)] = 1
        gtraces[0].data = y
        merged_traces.append(gtraces[0])
        assert np.all(has_filled), "some values not filled in!"
    return obspy.Stream(traces=merged_traces)


def _get_dtype(trace_list: List[trace_sequence]) -> np.dtype:
    """
    Return the datatype that should be used for the merged trace list.
    """
    # Try to determine datatype. Give preference to floats over ints
    used_types = {x.data.dtype for x in trace_list}
    float_used = used_types & NUMPY_FLOAT_TYPES
    int_used = used_types & NUMPY_INT_TYPES
    # if all else fails assume float32
    dtype = list(float_used or int_used) or list(np.float64)
    return dtype[0]


def stream2contiguous(stream: Stream) -> Stream:
    """
    Yields trimmed streams for times which all traces have data.

    Parameters
    ----------
    stream
        The input stream

    Examples
    --------
    >>> import obspy
    >>> st = obspy.read()
    >>> t1, t2 = st[0].stats.starttime, st[0].stats.endtime
    >>> _ = st[0].trim(endtime=t2 - 2)  # remove data at end of one trace
    >>> out = stream2contiguous(st)
    >>> # stream2contiguous should now have trimmed all traces to match
    >>> assert all(len(tr.data) for tr in st)
    """
    # pre-process waveforms by combining overlaps then breaking up masks
    stream.merge(method=1)
    stream = stream.split()
    # get seed_ids, start time, end time, and gaps
    seed_ids = {tr.id for tr in stream}
    starts, ends = _get_start_end(stream)
    # iterate start/end times, skip gaps and yield chunks of the waveforms
    for t1, t2 in zip(starts, ends):
        if t1 > t2 and len(starts) == len(ends) == 1:
            return  # if disjointed shutdown generator
        assert t1 < t2
        stream_out = stream.slice(starttime=to_utc(t1), endtime=to_utc(t2))
        stream_out.merge(method=1)
        if len({tr.id for tr in stream_out}) == len(seed_ids):
            yield stream_out


def _get_start_end(stream):
    """
    Get the start and end times of each contiguous chunk, return two lists
    that can be zipped and iterated.
    """
    starts = [max([tr.stats.starttime for tr in stream]).timestamp]
    ends = [min([tr.stats.endtime for tr in stream]).timestamp]
    gaps = stream.get_gaps(min_gap=0.01)
    # deal with gaps if there are any
    if len(gaps):
        df = _make_gap_dataframe(gaps)
        # group gaps together
        starts, ends = _get_stream_start_end(stream, df)
        gap_groups = df.groupby((df["t2"].shift() < df.t1).cumsum())
        t1_min = gap_groups.t1.min()
        t2_max = gap_groups.t2.max()
        starts = np.concatenate([[starts], t2_max.values])
        ends = np.concatenate([t1_min.values, [ends]])
    return starts, ends


def _get_stream_start_end(stream, gap_df):
    """
    Return a list of the latest start time of initial chunk and earliest
    endtime of last time chunk.
    """
    st1 = stream.slice(endtime=to_utc(gap_df.t1.min()))
    st2 = stream.slice(starttime=to_utc(gap_df.t2.max()))
    t1 = max([tr.stats.starttime.timestamp for tr in st1])
    t2 = min([tr.stats.endtime.timestamp for tr in st2])
    assert t1 < t2
    return t1, t2


def _make_gap_dataframe(gaps):
    """make a dataframe out of the gaps in a waveforms"""
    # get a dataframe of gaps
    columns = [
        "network",
        "station",
        "location",
        "channel",
        "starttime",
        "endtime",
        "duration",
        "samples",
    ]
    df = pd.DataFrame(gaps, columns=columns)
    df["t1"] = df.starttime.apply(lambda x: x.timestamp)
    df["t2"] = df.endtime.apply(lambda x: x.timestamp)
    df.sort_values("starttime", inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def _get_waveforms_bulk_naive(self, bulk_arg):
    """a naive implementation of get_waveforms_bulk that uses iteration."""
    st = obspy.Stream()
    for arg in bulk_arg:
        st += self.get_waveforms(*arg)
    return st


def archive_to_sds(
    bank: Union[Path, str, "obsplus.WaveBank"],
    sds_path: Union[Path, str],
    starttime: Optional[UTCDateTime] = None,
    endtime: Optional[UTCDateTime] = None,
    overlap: float = 30,
    type_code: str = "D",
    stream_processor: Optional[callable] = None,
):
    """
    Create a seiscomp data structure archive from a waveform source.

    Parameters
    ----------
    bank
        A wavebank or path to such.
    sds_path
        The path for the new sds archive to be created.
    starttime
        If not None, the starttime to convert data from bank.
    endtime
        If not None, the endtime to convert data from bank.
    overlap
        The overlap to use for each file.
    type_code
        The str indicating the datatype.
    stream_processor
        A callable that will take a single stream as input and return a
        a single stream. May return and empty stream to skip a stream.

    Notes
    -----
    see: https://www.seiscomp3.org/doc/applications/slarchive/SDS.html
    """
    sds_path = Path(sds_path)
    # create a fetcher object for yielding continuous waveforms
    bank = obsplus.WaveBank(bank)
    bank.update_index()
    # get starttime/endtimes
    index = bank.read_index()
    ts1 = index.starttime.min() if not starttime else starttime
    t1 = _nearest_day(ts1)
    t2 = to_utc(index.endtime.max() if not endtime else endtime)
    nslcs = get_seed_id_series(index).unique()
    # iterate over nslc and get data for selected channel
    for nslc in nslcs:
        nslc_dict = {n: v for n, v in zip(NSLC, nslc.split("."))}
        # yield waveforms in desired chunks
        ykwargs = dict(starttime=t1, endtime=t2, overlap=overlap, duration=86400)
        ykwargs.update(nslc_dict)
        for st in bank.yield_waveforms(**ykwargs):
            if stream_processor:  # apply stream processor if needed.
                st = stream_processor(st)
            if st:
                path = _get_sds_filename(st, sds_path, type_code, **nslc_dict)
                st.write(str(path), "mseed")


def _get_sds_filename(st, base_path, type_code, network, station, location, channel):
    """Given a stream get the expected path for the file."""
    time = _nearest_day(min([x.stats.starttime for x in st]))

    # add year and julday to formatting dict
    year, julday = "%04d" % time.year, "%03d" % time.julday
    filename = f"{network}.{station}.{location}.{channel}.{type_code}.{year}.{julday}"
    spath = f"{year}/{network}/{station}/{channel}.{type_code}"
    path = base_path / spath
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def _nearest_day(time):
    """Round a time down to the nearest day."""
    ts = to_utc(time).timestamp
    ts_day = 3600 * 24
    return to_utc(ts - (ts % ts_day))


@singledispatch
def get_waveform_client(waveforms: waveform_clientable_type) -> WaveformClient:
    """
    Extract a waveform client from various inputs.

    Parameters
    ----------
    waveforms
        Any of the following:
            * A path to an obspy-readable waveform file
            * A path to a directory of obspy-readable waveform files
            * A `obspy.Stream` instance
            * An instance of :class:`~obsplus.WaveBank`
            * Any other object that has a `get_waveforms` method

    Raises
    ------
    TypeError
        If a waveform client cannot be determined from the input.

    Notes
    -----
    If the output does not define a `get_waveform_bulk` method one will be added.
    """
    if not isinstance(waveforms, WaveformClient):
        msg = f"a waveform client could not be extracted from {waveforms}"
        raise TypeError(msg)
    # add waveform_bulk method dynamically if it doesn't exist already
    if not hasattr(waveforms, "get_waveforms_bulk"):
        bound_method = _get_waveforms_bulk_naive.__get__(waveforms)
        setattr(waveforms, "get_waveforms_bulk", bound_method)

    return waveforms


@get_waveform_client.register(str)
@get_waveform_client.register(Path)
def _get_waveclient_from_path(path):
    """get a waveform client from a path."""
    path = Path(path)
    if path.is_dir():
        return get_waveform_client(obsplus.WaveBank(path))
    else:
        return get_waveform_client(obspy.read(str(path)))


@singledispatch
def get_waveform_bulk_df(
    bulk,
    # bulk: Union[Sequence[waveform_request_type], pd.DataFrame]
) -> pd.DataFrame:
    """
    Create a bulk request dataframe from a variety of inputs.

    Parameters
    ----------
    bulk
        A sequence of
            [network, station, location, channel, starttime, endtime]
        or a dataframe with columns named the same.

    Returns
    -------
    A dataframe with waveform bulk columns.
    """
    df = pd.DataFrame(bulk, columns=list(WAVEFORM_REQUEST_DTYPES))
    return _df_to_waveform_bulk(df)


@get_waveform_bulk_df.register(pd.DataFrame)
def _df_to_waveform_bulk(df):
    """Ensure the dataframe has appropriate columns and return."""
    current_columns = set(df.columns)
    required_columns = list(WAVEFORM_REQUEST_DTYPES)
    # ensure columns exist
    if not current_columns.issuperset(required_columns):
        missing = set(required_columns) - set(current_columns)
        msg = (
            f"Dataframe is missing the following columns to be valid input"
            f" for bulk waveform request {missing}"
        )
        raise ValidationError(msg)
    out = df[list(required_columns)]
    return cast_dtypes(out, dtype=WAVEFORM_REQUEST_DTYPES)[required_columns]


def _filter_index_to_bulk(time, index_df, bulk_df) -> pd.DataFrame:
    """
    Using an index_df, apply conditions in request_df and return array indicating
    if values in index meet requested conditions.

    Parameters
    ----------
    time
        A tuple of mintime, maxtime
    index_df
        A dataframe indexing a waveform resource. Can be an index of traces
        in a stream or an index from a wavebank.
    bulk_df
        The dataframe containing bulk requests.
    """
    match_chars = {"*", "?", "[", "]"}
    # filter out any index times not in current time pair
    too_late = index_df["starttime"] > time[1]
    too_early = index_df["endtime"] < time[0]
    index_df = index_df[~(too_early | too_late)]
    ar = np.ones(len(index_df))  # indices of ind to use to load data
    # filter out any request times which are not for the current time pair
    is_starttime = bulk_df["starttime"] == time[0]
    is_endtime = bulk_df["endtime"] == time[1]
    bulk_df = bulk_df[is_starttime & is_endtime]
    # determine which columns use matching. These must be handled separately.
    uses_matches = [_column_contains(bulk_df[x], match_chars) for x in NSLC]
    match_ar = np.array(uses_matches).any(axis=0)
    df_match = bulk_df[match_ar]
    df_no_match = bulk_df[~match_ar]
    # handle columns that need matches (more expensive)
    if not df_match.empty:
        match_bulk = df_match.to_records(index=False)
        mar = np.array([filter_index(index_df, *tuple(b)[:4]) for b in match_bulk])
        ar = np.logical_and(ar, mar.any(axis=0))
    # handle columns that do not need matches
    if not df_no_match.empty:
        nslc1 = set(get_seed_id_series(df_no_match))
        nslc2 = get_seed_id_series(index_df)
        ar = np.logical_and(ar, nslc2.isin(nslc1))
    # get a list of used traces, combine and trim
    return index_df[ar]
