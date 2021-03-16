"""
Module for adding the client-like "get_waveforms" to the Stream class
"""
from typing import Optional

import numpy as np
import obspy
from obspy import Stream, UTCDateTime as UTC

from obsplus.constants import SMALLDT64, LARGEDT64, NSLC, bulk_waveform_arg_type
from obsplus.utils.time import to_utc
from obsplus.utils.waveforms import (
    _get_waveform_df,
    get_waveform_bulk_df,
    _filter_index_to_bulk,
    merge_traces,
)


def get_waveforms(
    stream: Stream,
    network: str = "*",
    station: str = "*",
    location: str = "*",
    channel: str = "*",
    starttime: Optional[UTC] = None,
    endtime: Optional[UTC] = None,
) -> obspy.Stream:
    """
    A subset of the Client.get_waveforms method.

    Simply makes successive calls to Stream.select and Stream.trim under the
    hood. Matching is available on all str parameters.

    Parameters
    ----------
    stream
        A stream object.
    network
        The network code
    station
        The station code
    location
        Location code
    channel
        Channel code
    starttime
        Starttime for query
    endtime
        Endtime for query
    """
    t1, t2 = to_utc(starttime or SMALLDT64), to_utc(endtime or LARGEDT64)
    kwargs = {c: v for c, v in zip(NSLC, [network, station, location, channel])}
    st = stream.select(**kwargs).slice(starttime=t1, endtime=t2).copy()
    return st


def get_waveforms_bulk(
    stream: Stream, bulk: bulk_waveform_arg_type, **kwargs
) -> Stream:
    """
    Get a large number of waveforms with a bulk request.

    Parameters
    ----------
    stream
        A stream object.
    bulk
        A list of any number of tuples containing the following:
        (network, station, location, channel, starttime, endtime).
    """
    # get a dataframe of stream contents
    index = _get_waveform_df(stream)
    # get a dataframe of the bulk arguments, convert time to datetime64
    request_df = get_waveform_bulk_df(bulk)
    if not len(request_df):  # return empty string if no bulk reqs provided
        return obspy.Stream()
    # get unique times and check conditions for string columns
    unique_times = np.unique(request_df[["starttime", "endtime"]].values, axis=0)
    traces = []
    for (t1, t2) in unique_times:
        sub = _filter_index_to_bulk((t1, t2), index_df=index, bulk_df=request_df)
        new = obspy.Stream(traces=[x.data for x in sub["trace"]]).slice(
            starttime=to_utc(t1), endtime=to_utc(t2)
        )
        traces.extend(new.traces)
    return merge_traces(obspy.Stream(traces=traces))


# --- add get_waveforms to Stream class

obspy.Stream.get_waveforms = get_waveforms
obspy.Stream.get_waveforms_bulk = get_waveforms_bulk
