"""
Waveframe class definition.
"""
import obspy
import pandas as pd

from obsplus.constants import waveform_clientable_type, BULK_WAVEFORM_COLUMNS
from obsplus.structures.waveframe.utils import _time_to_utc
from obsplus.utils import order_columns


class WaveFrame:
    """
    A dataframe-like class for representing waveforms and associated data.
    """

    def __init__(self, waveform_client: waveform_clientable_type, stats: pd.DataFrame):
        # first get bulk request arguments
        stats = _time_to_utc(order_columns(stats, BULK_WAVEFORM_COLUMNS))
        nslc = stats[list(BULK_WAVEFORM_COLUMNS)].to_records(index=False).tolist()
        # st = waveform_client.get_bulk_waveforms(nslc)

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
        return cls(waveform_client=stream, stats=stats)
