"""
Waveframe class definition.
"""
import obspy
import pandas as pd

from obsplus.constants import waveform_clientable_type
from obsplus.structures.waveframe.utils import (
    _get_waveforms_bulk,
    _get_bulk_args,
    _get_stats_dataframe,
    _get_data_array_from_client,
)
from obsplus.waveforms.utils import stream_bulk_split


class WaveFrame:
    """
    A dataframe-like class for representing waveforms and associated data.

    Parameters
    ----------
    waveforms
        Any object with a ``get_waveforms`` method, and, optionally, a
        ``get_waveforms_bulk`` method.
    stats
        A dataframe with at least the following columns:
            (network, station, location, channel, starttime, endtime)
        Wildcards in any of the string columns are not permitted.
    """

    def __init__(self, waveforms: waveform_clientable_type, stats: pd.DataFrame):
        # validate stats dataframe and extract bulk parameters
        self.stats = _get_stats_dataframe(stats)
        bulk = _get_bulk_args(self.stats)
        # make sure we have a stream
        array = _get_data_array_from_client(waveforms, bulk)

        # breakpoint()

    # --- Alternative constructors
    @classmethod
    def from_stream(cls, stream: obspy.Stream) -> "WaveFrame":
        """
        Get a WaveFrame from an ObsPy stream.

        Parameters
        ----------
        stream
            An obspy Stream.

        Notes
        -----
        It is best to ensure the traces have data which are about the same
        size since dataframes must be square.
        """
        stats = pd.DataFrame([dict(tr.stats) for tr in stream])
        return cls(waveforms=stream, stats=stats)

    def to_stream(self) -> obspy.Stream:
        """
        Convert the waveframe to a Stream object.
        """
        return obspy.Stream()
