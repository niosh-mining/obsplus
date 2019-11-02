"""
Waveframe class definition.
"""
import obspy
import copy
import pandas as pd
from typing import Optional, Union

from obsplus.constants import waveform_clientable_type
from obsplus.utils.waveframe import (
    _get_bulk_args,
    _get_stats_dataframe,
    _get_timeseries_df_from_client,
    DfPartDescriptor,
)
from obsplus.utils.time import to_utc


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

    _df: pd.DataFrame
    stats = DfPartDescriptor("_df")
    data = DfPartDescriptor("_df")

    def __init__(
        self,
        stats: Union[pd.DataFrame, "WaveFrame"],
        waveforms: Optional[waveform_clientable_type] = None,
    ):
        self.processing = []  # init empty processing list

        if isinstance(stats, WaveFrame):
            df = stats._df.copy()
            self.processing = copy.deepcopy(stats.processing)
        else:
            df = self._df_from_stats_waveforms(stats=stats, waveforms=waveforms)
        self._df = df

    def _df_from_stats_waveforms(self, stats, waveforms):
        """ Get the waveframe df from stats and waveform_client. """
        # validate stats dataframe and extract bulk parameters
        stats = _get_stats_dataframe(stats)
        bulk = _get_bulk_args(stats)
        # make sure we have a stream
        ts_df = _get_timeseries_df_from_client(waveforms, bulk)
        return pd.concat([stats, ts_df], axis=1, keys=["stats", "data"])

    def __str__(self):
        return str(self._df)

    def __repr__(self):
        return repr(self._df)

    def __len__(self):
        return len(self._df)

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

    # --- Output
    def to_stream(self) -> obspy.Stream:
        """
        Convert the waveframe to a Stream object.
        """
        traces = []
        # get stats, convert datetimes back to obspy
        stats = self._df["stats"].copy()
        for col in ["starttime", "endtime"]:
            stats[col] = to_utc(stats[col])
        # create traces
        for ind, row in stats.iterrows():
            stats = row.to_dict()
            data = self._df.loc[ind, "data"]
            traces.append(obspy.Trace(data=data.values, header=stats))
        return obspy.Stream(traces=traces)
