"""
Waveframe class definition.
"""
import copy
import operator
from typing import Optional, Union, Any, List

import numpy as np
import obspy
import pandas as pd

from obsplus.constants import (
    waveform_clientable_type,
    number_type,
    WAVEFRAME_STATS_DTYPES,
)
from obsplus.exceptions import IncompatibleWaveFrameError
from obsplus.utils.validate import validator, validate
from obsplus.utils.waveframe import (
    _DfPartDescriptor,
    _DataStridder,
    _stats_df_to_stats,
    _DFExtractorFromStatsAndClient,
    _update_df_times,
)


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
    processing
        A list of strings indicating the methods called on the waveframe.
    """

    _df: pd.DataFrame
    stats = _DfPartDescriptor("_df")
    data = _DfPartDescriptor("_df")
    _required_stats_columns = set(sorted(WAVEFRAME_STATS_DTYPES))

    def __init__(
        self,
        stats: Union[pd.DataFrame, "WaveFrame"],
        waveforms: Optional[waveform_clientable_type] = None,
        processing: Optional[List[str]] = None,
    ):
        self.processing = processing or []  # init empty processing list
        # for an instance of WaveFrame copy and update state
        if isinstance(stats, WaveFrame):
            self.__dict__.update(stats.copy().__dict__)
            return
        # the full waveframe dataframe was passed
        elif isinstance(stats, pd.DataFrame) and waveforms is None:
            df = stats.copy()
        # waveform argument is already a dataframe
        elif isinstance(waveforms, pd.DataFrame):
            df = pd.concat([stats, waveforms], axis=1, keys=["stats", "data"])
        # else we have
        else:
            extractor = _DFExtractorFromStatsAndClient(stats, waveforms)
            df = extractor.get_df()
        # the waveframe must have unique indicies
        assert df.index.is_unique
        self._df = df

    def __str__(self):
        return str(self._df)

    def __repr__(self):
        return repr(self._df)

    def __len__(self):
        return len(self._df)

    def __add__(self, other):
        return self.operate(other, operator.add)

    def __sub__(self, other):
        return self.operate(other, operator.sub)

    def __mul__(self, other):
        return self.operate(other, operator.mul)

    def __truediv__(self, other):
        return self.operate(other, operator.truediv)

    def __eq__(self, other):
        return self.equals(other)

    def __getitem__(self, item):
        try:
            return self._df[("stats", item)]
        except TypeError:  # boolean index is being used
            return WaveFrame(self._df[item], processing=self.processing)

    def __setitem__(self, key, value):
        self._df[("stats", key)] = value

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return self.data.shape

    @property
    def index(self):
        return self._df.index

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

    def from_data(self, data: Union[pd.DataFrame, np.ndarray]) -> "WaveFrame":
        """
        Return a copy of the current waveframe with a new data df.

        Parameters
        ----------
        data
            A dataframe with the same index as the current stats dataframe.
        """
        out = self.copy()
        if isinstance(data, np.ndarray):
            old = self.data
            data = pd.DataFrame(data, columns=old.columns, index=old.index)
        out.data = data
        return out

    def from_stats(self, stats: pd.DataFrame) -> "WaveFrame":
        """
        Return a copy of the current waveframe with a new stats df.

        Parameters
        ----------
        stats
            A dataframe with the same columns as the stats df.
        """
        out = self.copy()
        out.stats = stats
        return out

    # --- Concrete implementations of dunder methods
    def copy(self) -> "WaveFrame":
        """
        Perform a deep copy on the waveframe.
        """
        return copy.deepcopy(self)

    def _stats_equal(self, other: "WaveFrame", stats_intersection=False) -> bool:
        """
        Return True if the stats on self and other are equal.

        Parameters
        ----------
        other
            Another WaveFrame
        stats_intersection
            If True only check the intersection of columns for stats.
        """
        stats1, stats2 = self.stats, other.stats
        if stats_intersection:
            cols = list(set(stats1) & set(stats2))
            stats1, stats2 = stats1[cols], stats2[cols]
        if not stats1.equals(stats2):
            return False
        return True

    def equals(
        self,
        other: Union[Any, "WaveFrame"],
        processing: bool = False,
        stats_intersection: bool = False,
    ) -> bool:
        """
        Equality checks for an WaveFrame.

        Returns True if both the data and stats dataframes are equal.

        Parameters
        ----------
        other
            An object against which equality will be checked. If obj is not
            an instance of waveframe the result will be False.
        processing
            If True, include the processing list in the equality check.
        stats_intersection
            If True, only compare the intersections of columns from both
            waveframes' stats dataframe.
        """
        if not isinstance(other, WaveFrame):
            return False
        # check if stats are equal
        if not self._stats_equal(other, stats_intersection=stats_intersection):
            return False
        # check if data are equal
        data1, data2 = self.data, other.data
        if not data1.equals(data2):
            return False
        # check processing, if requested
        if processing:
            if not self.processing == other.processing:
                return False
        return True

    def operate(
        self, other: number_type, operator: callable, check_stats: bool = True
    ) -> "WaveFrame":
        """
        Perform a basic operation on a WaveFrame.

        Parameters
        ----------
        other
            Either a number, in which case it will be added to all values in
            the data df, or another waveframe, in which case the data dfs will
            simply be added together.

        operator
            A callable which will take (self.data, other).
        """
        if isinstance(other, WaveFrame):
            if check_stats and not self._stats_equal(other, stats_intersection=True):
                msg = "Stats columns are not equal, cannot add waveframes."
                raise IncompatibleWaveFrameError(msg)
            other = other.data
        new_data = operator(self.data.values, other)
        return self.from_data(new_data)

    # --- Output
    def to_stream(self) -> obspy.Stream:
        """
        Convert the waveframe to a Stream object.
        """
        # get stats, convert datetimes back to obspy
        stats = _stats_df_to_stats(self.stats)
        # create traces
        traces = []
        for ind, row in stats.iterrows():
            stats = row.to_dict()
            data = self._df.loc[ind, "data"]
            traces.append(obspy.Trace(data=data.values, header=stats))
        return obspy.Stream(traces=traces)

    # --- Utilities
    def validate(self, report=False) -> Union["WaveFrame", pd.DataFrame]:
        """
        Runs the basic WaveFrame validation suite, returns self if passes.

        Parameters
        ----------
        report
            If True, return the validation report, as a dataframe, rather than
            self.
        """
        out = validate(self, "ops_waveframe", report=report)
        if report:
            return out
        return self

    def dropna(
        self, axis: int = 0, how: str = "any", thresh: Optional[float] = None
    ) -> "WaveFrame":
        """
        Drop data with null values, return a new waveframe.

        Parameters
        ----------
        axis
        how: {'any', 'all'}
            The string for determining if a row or column is deleted.
            'any': Drop row or column if any null values are present
            'all': Drop row or column if all null values are present
        thresh
        """
        data = self.data
        new_data = data.dropna(axis=axis, how=how, thresh=thresh)
        stats = self.stats.loc[new_data.index]
        df = pd.concat([stats, new_data], axis=1, keys=["stats", "data"])
        df = _update_df_times(df)
        return WaveFrame(df)

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
        out = _DataStridder(self._df).stride(window_len, overlap)
        return WaveFrame(out)


# --- WaveFrame validators


@validator("ops_waveframe", WaveFrame)
def has_stats_and_data(wf: WaveFrame):
    """ Ensure the waveframe has the required stats columns. """
    df = wf._df
    levels = df.columns.get_level_values(0).unique()
    assert {"data", "stats"}.issubset(levels)


@validator("ops_waveframe", WaveFrame)
def has_required_stats_columns(wf: WaveFrame):
    """ Ensure the waveframe has the required stats columns. """
    current_cols = wf.stats.columns
    expected_cols = wf._required_stats_columns
    assert set(current_cols).issuperset(expected_cols)


@validator("ops_waveframe", WaveFrame)
def starttime_le_endtime(wf):
    """ ensure starttimes is less than the endtime. """
    assert (wf["starttime"] <= wf["endtime"]).all()


@validator("ops_waveframe", WaveFrame)
def times_consistent_with_data_len(wf):
    """ Ensure the start and endtimes are consistent with the data length. """
    data_len = wf.shape[-1]
    expected_duration = wf["delta"] * (data_len - 1)
    assert (wf["starttime"] + expected_duration == wf["endtime"]).all()
