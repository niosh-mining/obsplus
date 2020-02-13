"""
Module for adding the client-like "get_waveforms" to the Stream class
"""
from functools import reduce
from operator import add
from typing import Optional

import numpy as np
import obspy
import pandas as pd
from obspy import Stream, UTCDateTime as UTC

from obsplus.constants import SMALLDT64, LARGEDT64, NSLC, bulk_waveform_arg_type
from obsplus.utils.pd import filter_index, _column_contains
from obsplus.utils.pd import get_seed_id_series
from obsplus.utils.time import to_datetime64
from obsplus.utils.time import to_utc
from obsplus.utils.waveforms import _stream_data_to_df


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
    if not bulk:  # return emtpy waveforms if empty list or None
        return obspy.Stream()

    def _func(time, ind, df, st):
        """ return waveforms from df of bulk parameters """
        match_chars = {"*", "?", "[", "]"}
        ar = np.ones(len(ind))  # indices of ind to use to load data
        _t1, _t2 = time[0], time[1]
        df = df[(df.t1 == time[0]) & (df.t2 == time[1])]
        # determine which columns use any matching or other select features
        uses_matches = [_column_contains(df[x], match_chars) for x in NSLC]
        match_ar = np.array(uses_matches).any(axis=0)
        df_match = df[match_ar]
        df_no_match = df[~match_ar]
        # handle columns that need matches (more expensive)
        if not df_match.empty:
            match_bulk = df_match.to_records(index=False)
            mar = np.array([filter_index(ind, *tuple(b)[:4]) for b in match_bulk])
            ar = np.logical_and(ar, mar.any(axis=0))
        # handle columns that do not need matches
        if not df_no_match.empty:
            nslc1 = set(get_seed_id_series(df_no_match))
            nslc2 = get_seed_id_series(ind)
            ar = np.logical_and(ar, nslc2.isin(nslc1))
        # get a list of used traces, combine and trim
        st = obspy.Stream([x for x, y in zip(st, ar) if y])
        return st.slice(starttime=to_utc(_t1), endtime=to_utc(_t2))

    # get a dataframe of stream contents
    index = _stream_data_to_df(stream)
    # get a dataframe of the bulk arguments, convert time to datetime64
    df = pd.DataFrame(bulk, columns=list(NSLC) + ["utc1", "utc2"])
    df["t1"] = df["utc1"].apply(to_datetime64)
    df["t2"] = df["utc2"].apply(to_datetime64)
    t1, t2 = df["t1"].min(), df["t2"].max()
    # filter index and streams to be as short as possible
    needed: pd.DataFrame = ~((index.starttime > t2) | (index.endtime < t1))
    ind = index[needed]
    stream = obspy.Stream([tr for tr, bo in zip(stream, needed.values) if bo])
    # groupby.apply calls two times for each time set, avoid this.
    unique_times = np.unique(df[["t1", "t2"]].values, axis=0)
    streams = [_func(time, df=df, ind=ind, st=stream) for time in unique_times]
    return reduce(add, streams)


# --- add get_waveforms to Stream class

obspy.Stream.get_waveforms = get_waveforms
obspy.Stream.get_waveforms_bulk = get_waveforms_bulk
