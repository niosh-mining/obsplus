"""
Module for adding the client-like "get_waveforms" to the Stream class
"""
import obspy
from obspy import Stream, UTCDateTime as UTC


def get_waveforms(
    stream: Stream,
    network: str,
    station: str,
    location: str,
    channel: str,
    starttime: UTC,
    endtime: UTC,
):
    """
    A subset of the Client.get_waveforms method.

    Matching is available on all str paramters.

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
    st = st.trim(starttime=UTC(starttime), endtime=UTC(endtime))
    return st


def get_waveforms_bulk(stream, bulk_args):
    """ get bulk waveforms from waveforms """
    st = obspy.Stream()
    for arg in bulk_args:
        st += get_waveforms(stream, *arg)
    return st


# --- add get_waveforms to Stream class

obspy.Stream.get_waveforms = get_waveforms
obspy.Stream.get_waveforms_bulk = get_waveforms_bulk
