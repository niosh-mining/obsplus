"""
Stream utilities
"""
import warnings
from functools import singledispatch
from pathlib import Path
from typing import Optional

import numpy as np
import obspy
import pandas as pd
from obspy import Stream, UTCDateTime

import obsplus
from obsplus.constants import waveform_clientable_type
from obsplus.interfaces import WaveformClient


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
    short trace is found

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
    """ get the starttimes and endtimes for trimming, raise ValueError
    if the waveforms is disjointed """
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
    t1, t2 = UTCDateTime(df.start.max()), UTCDateTime(df.end.min())
    if t2 < t1:
        msg = f"The following waveforms has traces with no overlaps {stream}"
        raise ValueError(msg)
    return stream.trim(starttime=t1, endtime=t2)


def stream2contiguous(stream: Stream):
    """ generator to yield contiguous streams from disjointed streams """
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
        stream_out = stream.slice(starttime=UTCDateTime(t1), endtime=UTCDateTime(t2))
        stream_out.merge(method=1)
        if len({tr.id for tr in stream_out}) == len(seed_ids):
            yield stream_out


def _get_start_end(stream):
    """ get the start and end times of each contiguous chunk,
     return two lists that can be zipped and iterated """
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
    """ return a list of the latest start time of initial chunk and earliest
    endtime of last time chunk """
    st1 = stream.slice(endtime=UTCDateTime(gap_df.t1.min()))
    st2 = stream.slice(starttime=UTCDateTime(gap_df.t2.max()))
    t1 = max([tr.stats.starttime.timestamp for tr in st1])
    t2 = min([tr.stats.endtime.timestamp for tr in st2])
    assert t1 < t2
    return t1, t2


def _make_gap_dataframe(gaps):
    """ make a dataframe out of the gaps in a waveforms"""
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
    """ a naive implementation of get_waveforms_bulk that uses iteration. """
    st = obspy.Stream()
    for arg in bulk_arg:
        st += self.get_waveforms(*arg)
    return st


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
    If the output does define a `get_waveform_bulk` method one will be added.
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
    """ get a waveform client from a path. """
    path = Path(path)
    if path.is_dir():
        return get_waveform_client(obsplus.WaveBank(path))
    else:
        return get_waveform_client(obspy.read(str(path)))
