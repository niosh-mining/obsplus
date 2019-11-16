"""
Waveframe class definition.
"""
import copy
from typing import Optional, Union

import obspy
import pandas as pd

from obsplus.constants import waveform_clientable_type
from obsplus.utils.time import to_utc
from obsplus.utils.waveframe import (
    _get_data_and_stats,
    _create_stats_df,
    DfPartDescriptor,
)
from obsplus.utils.pd import get_waveforms_bulk_args


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
        # for an instance of WaveFrame simply return its df
        if isinstance(stats, WaveFrame):
            df = stats._df.copy()
            self.processing = copy.deepcopy(stats.processing)
        # the full waveframe dataframe was passed
        elif isinstance(stats, pd.DataFrame) and waveforms is None:
            df = stats.copy()
        else:
            df = self._df_from_stats_waveforms(stats=stats, waveforms=waveforms)
        self._df = df

    def _df_from_stats_waveforms(self, stats, waveforms):
        """ Get the waveframe df from stats and waveform_client. """
        # validate stats dataframe and extract bulk parameters
        bulk = get_waveforms_bulk_args(stats)
        # get arrays and stats list
        data_list, stats_list = _get_data_and_stats(waveforms, bulk)
        ts_df = pd.DataFrame(data_list)
        # now augment stats df with data from stats traces
        stats = _create_stats_df(stats_list)
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

    # --- Utilities
    def stride(self, window_len: Optional[int] = None, overlap: int = 0) -> "WaveFrame":
        """
        Stride a waveframe to create more rows and fewer columns.

        Parameters
        ----------
        window_len
            The window length in samples.
        overlap
            The overlap between each waveform slice, in samples.
        """
        window_len = window_len or self.data.size[-1]
        assert overlap > window_len
