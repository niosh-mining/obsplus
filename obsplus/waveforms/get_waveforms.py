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
    _stream_data_to_df,
    get_waveform_bulk_df,
    _filter_index_to_bulk,
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
    index = _stream_data_to_df(stream)
    # get a dataframe of the bulk arguments, convert time to datetime64
    df = get_waveform_bulk_df(bulk)
    if not len(df):  # return empty string if no bulk reqs provided
        return obspy.Stream()
    # filter stream and index to only include requested times
    min_time, max_time = index["starttime"].min(), index["endtime"].max()
    needed = ~((index.starttime > max_time) | (index.endtime < min_time))
    index = index[needed]
    stream = obspy.Stream([tr for tr, bo in zip(stream, needed.values) if bo])
    # get unique times and check conditions for string columns
    # groupby.apply calls two times for each time set, avoid this.
    unique_times = np.unique(df[["t1", "t2"]].values, axis=0)
    out = obspy.Stream()
    for utime in unique_times:
        ar = _filter_index_to_bulk(utime, index_df=index, bulk_df=df)
        st = obspy.Stream([x for x, y in zip(stream, ar) if y])
        out += st.slice(starttime=to_utc(utime[0]), endtime=to_utc(utime[1]))
    return out


# --- add get_waveforms to Stream class

obspy.Stream.get_waveforms = get_waveforms
obspy.Stream.get_waveforms_bulk = get_waveforms_bulk
