"""
Module for adding the client-like "get_waveforms" to the Stream class
"""
from typing import Optional

import obspy
from obspy import Stream, UTCDateTime as UTC

from obsplus.constants import BIG_UTC, SMALL_UTC


def get_waveforms(
    stream: Stream,
    network: str = "*",
    station: str = "*",
    location: str = "*",
    channel: str = "*",
    starttime: Optional[UTC] = None,
    endtime: Optional[UTC] = None,
):
    """
    A subset of the Client.get_waveforms method.

    Simply makes successive calls to Stream.select and Stream.trim under the
    hood. Matching is available on all str parameters.

    Parameters
    ----------
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

    Returns
    -------
    Stream
    """
    stream = stream.copy()
    st = stream.select(
        network=network, station=station, location=location, channel=channel
    )
    st = st.trim(starttime=UTC(starttime or SMALL_UTC), endtime=UTC(endtime or BIG_UTC))
    return st


def get_waveforms_bulk(stream, bulk_args):
    """ get bulk waveforms from waveforms """
    st = obspy.Stream()
    for arg in bulk_args:
        stt = get_waveforms(stream, *arg)
        # if a stream was returned then add it to original
        if stt is not None:
            st += stt
    return st


# --- add get_waveforms to Stream class

obspy.Stream.get_waveforms = get_waveforms
obspy.Stream.get_waveforms_bulk = get_waveforms_bulk
