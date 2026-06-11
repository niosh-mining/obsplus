"""
Tools for creating and converting waveframes from various formats.
"""

import numpy as np
import obspy
import pandas as pd

import obsplus
from obsplus.constants import (
    bulk_waveform_arg_type,
    WAVEFRAME_STATS_DTYPES,
    BULK_WAVEFORM_COLUMNS,
)
from obsplus.interfaces import WaveformClient
from obsplus.utils.misc import register_func
from obsplus.utils.pd import (
    cast_dtypes,
    order_columns,
    get_waveforms_bulk_args,
    apply_funcs_to_columns,
)
from obsplus.utils.waveforms import stream_bulk_split
from obsplus.waveframe.core import _combine_stats_and_data


class _WFExampleLoader:
    """
    Private class for loading the example waveframes.
    """

    example_loaders = {}
    cache = {}

    def __call__(self, name: str, *args, **kwargs):
        # load cached example and return copy
        if name in self.cache:
            return self.cache[name].copy()
        # or create it if it does not yet exist
        self.cache[name] = self.example_loaders[name](self, *args, **kwargs)
        return self(name)

    @register_func(example_loaders, "default")
    def _make_wf_from_st_no_response(self):
        """Get a copy of the default trace, remove response."""
        st = obspy.read()
        # drop response for easier stats dtypes
        for tr in st:
            tr.stats.pop("response", None)
        return obsplus.WaveFrame.from_stream(st)


class _DFExtractorFromStatsAndClient:
    """
    Private class for extracting info from stats and waveform clients.
    """

    def __init__(self, stats, waveforms):
        self.stats = stats
        self.client = obsplus.get_waveform_client(waveforms)

    def _get_waveforms_bulk(
        self, waveform_client: WaveformClient, bulk: bulk_waveform_arg_type
    ) -> obspy.Stream:
        """
        Get bulk waveforms from waveform client.

        Either 1) call waveform_client's get_waveforms_bulk method (if it exists)
        or 2) iterate the bulk_arg and call get_waveforms for each.

        Parameters
        ----------
        waveform_client
            Any object with a ``get_waveforms`` method.
        bulk
            A list of tuples containing:
                (network, station, location, channel, starttime, endtime)
        """
        if hasattr(waveform_client, "get_waveforms_bulk"):
            return waveform_client.get_waveforms_bulk(bulk)
        # iterate each bulk, collect streams and return
        out = obspy.Stream()
        for bulk_arg in bulk:
            out += waveform_client.get_waveforms(*bulk_arg)
        return out

    def _create_stats_df(self, df, strip_extra=True) -> pd.DataFrame:
        """
        Create a stats dataframe from a list of trace stats objects.

        Parameters
        ----------
        stats_list
            A list of stats objects.
        strip_extra
            If True, strip out columns called "processing" and "response" if
            found.
        """
        out = df.pipe(cast_dtypes, WAVEFRAME_STATS_DTYPES).pipe(
            order_columns, list(WAVEFRAME_STATS_DTYPES)
        )
        # strip extra columns that can have complex object types
        if strip_extra:
            to_drop = ["processing", "response"]
            drop = list(set(to_drop) & set(df.columns))
            out = out.drop(columns=drop)
        return out

    def _get_data_df(self, arrays):
        """A fast way to convert a list of np.ndarrays to a single ndarray."""
        # Surprisingly this is much (5x) times faster than just passing arrays
        # to the DataFrame constructor.
        max_len = max([len(x) if x.shape else 0 for x in arrays])
        out = np.full([len(arrays), max_len], np.NAN)
        for num, array in enumerate(arrays):
            if not array.shape:
                continue
            out[num, : len(array)] = array
        # out has an empty dim just use NaN
        if out.shape[-1] == 0:
            return pd.DataFrame([np.NaN])
        return pd.DataFrame(out)

    def _get_data_and_stats(self, bulk):
        """Using a waveform client return an array of data and stats."""
        waveforms = self.client
        # Get a stream of waveforms.
        if not isinstance(waveforms, obspy.Stream):
            waveforms = self._get_waveforms_bulk(waveforms, bulk)
        # There isn't guaranteed to be a trace for each bulk arg, so use
        # stream_bulk_split to make it so.
        st_list = stream_bulk_split(waveforms, bulk, fill_value=np.NaN)
        # make sure the data are merged together with a sensible fill value
        arrays, stats = [], []
        for st, b in zip(st_list, bulk):
            assert len(st) in {0, 1}, "st should either be empty or len 1"
            if not len(st):  # empty data still needs an entry and stats from bulk
                arrays.append(np.array(np.NaN))
                statskwargs = {i: v for i, v in zip(BULK_WAVEFORM_COLUMNS, b)}
                stats.append(statskwargs)
                continue
            arrays.append(st[0].data)
            stats.append(dict(st[0].stats))

        return arrays, stats

    def get_df(self):
        """
        Return the input dataframe for WaveFrame.
        """
        # validate stats dataframe and extract bulk parameters
        bulk = get_waveforms_bulk_args(self.stats)
        # get arrays and stats list
        data_list, stats_list = self._get_data_and_stats(bulk)
        # then get dataframes of stats and arrays
        stats = self._create_stats_df(pd.DataFrame(stats_list))
        data = self._get_data_df(data_list)
        return _combine_stats_and_data(stats, data)


class _DFtoStreamConverter:
    """Class for converting dataframes to and from streams."""

    def _stats_df_to_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare stats dataframe to be converted back to stats objects.

        Returns new df with obspy datatypes.
        """
        # get inputs to cast_dtypes, apply_funcs, drop, etc.
        dtypes = {"starttime": "utcdatetime", "endtime": "utcdatetime"}
        funcs = {"delta": lambda x: x.astype(int) / 1_000_000_000}
        drop = {"sampling_rate"} & set(df.columns)

        df = (
            df.pipe(cast_dtypes, dtypes)
            .pipe(apply_funcs_to_columns, funcs=funcs)
            .drop(columns=drop)
        )
        return df

    def __call__(self, df):
        """convert waveframe df to stream."""
        stats_df_old, data_df = df["stats"], df["data"]
        # get stats, convert datetimes back to obspy
        stats_df = self._stats_df_to_stats(stats_df_old)
        # create traces
        traces = []
        for ind, row in stats_df.iterrows():
            stats = row.to_dict()
            data = data_df.loc[ind]
            traces.append(obspy.Trace(data=data.values, header=stats))
        return obspy.Stream(traces=traces)
