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
    trim_time_types,
    starttime_endtime_params,
)
from obsplus.exceptions import IncompatibleWaveFrameError
from obsplus.utils.validate import validator, validate
from obsplus.utils.docs import compose_docstring
from obsplus.waveframe.construct import (
    _WFExampleLoader,
    _DFExtractorFromStatsAndClient,
    _DFtoStreamConverter,
)
from obsplus.waveframe.core import _update_df_times, _combine_stats_and_data
from obsplus.waveframe.processing import _WFDetrender
from obsplus.waveframe.reshape import _DataStridder, _NaNHandler, _Trimmer


class _DfPartDescriptor:
    """
    A simple descriptor granting access to various parts of a dataframe.
    """

    def __init__(self, df_name):
        self._df_name = df_name

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        if instance is None:
            msg = f"{self.name} is an only an instance attribute"
            raise AttributeError(msg)
        return getattr(instance, self._df_name)[self._name]

    def __set__(self, instance, value):
        # for some reason this was needed to prevent overwriting the property
        msg = f"can't set attribute {self._name}"
        raise AttributeError(msg)


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
    # a tuple of stats attributes which should be treated as read only
    _read_only_stats = "endtime"

    def __init__(
        self,
        stats: Union[pd.DataFrame, "WaveFrame"],
        waveforms: Optional[Union[waveform_clientable_type, pd.DataFrame]] = None,
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
            df = _combine_stats_and_data(stats, waveforms)
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
        except KeyError:
            cols = str(self.stats.columns)
            msg = (
                f"{item} is not a stats column of this waveframe."
                f" Columns are:\n {cols}"
            )
            raise KeyError(msg)

    def __setitem__(self, key, value):
        if key in self._read_only_stats:
            msg = f"{key} is a read-only stats column"
            raise AttributeError(msg)
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
        if isinstance(data, np.ndarray):
            old = self.data
            data = pd.DataFrame(data, columns=old.columns, index=old.index)
        return WaveFrame(stats=self.stats, waveforms=data)

    def from_stats(self, stats: pd.DataFrame) -> "WaveFrame":
        """
        Return a copy of the current waveframe with a new stats df.
        Parameters
        ----------
        stats
            A dataframe with the same columns as the stats df.
        """
        data = self.data
        return WaveFrame(stats=stats, waveforms=data)

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
        converter = _DFtoStreamConverter()
        return converter(self._df)

    # --- Utilities
    @classmethod
    def load_example_wf(cls, name: str = "default", *args, **kwargs) -> "WaveFrame":
        """
        Load an example WaveFrame.

        Args and kwargs are passed to the example generating functions.

        Parameters
        ----------
        name
            The name of the example to load. By default a waveframe will
            be created from he obspy example traces.

        Examples
        --------
        Get the default waveframe.

        >>> wf = WaveFrame.load_example_wf()
        >>> assert isinstance(wf, WaveFrame)
        >>> assert not wf.empty
        """
        example_loader = _WFExampleLoader()
        return example_loader(name, *args, **kwargs)

    def validate(self, report=False) -> Union["WaveFrame", pd.DataFrame]:
        """
        Runs the basic WaveFrame validation suite.

        This method does the following:
            1. Ensure the underlying dataframe has both stats and data

        Parameters
        ----------
        report
            If True, return the validation report, as a dataframe, rather than
            self or raising assertion errors if validators fail.
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
        nan_hand = _NaNHandler("dropna")
        df = nan_hand(self._df, axis=axis, how=how, thresh=thresh)

        return WaveFrame(df)

    def fillna(self, value=None, method=None, axis=None, limit=None):
        """
        Fill NaN values in data.

        Parameters
        ----------
        value
            The fill value.
        method: {'backfill', 'pad' bfill', ffill, None}
            Method used for filling hoes in reindexing series.
        axis
            Axis along which to fill missing values.
        limit
            Limit to number of cconsecutive NaN values.
        """
        nan_hand = _NaNHandler("fillna")
        kwargs = dict(value=value, method=method, axis=axis, limit=limit)
        df = nan_hand(self._df, **kwargs)
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
        out = _DataStridder()(self._df, window_len=window_len, overlap=overlap)
        return WaveFrame(out)

    @compose_docstring(start_end_desc=starttime_endtime_params)
    def trim(
        self,
        starttime: Optional[trim_time_types] = None,
        endtime: Optional[trim_time_types] = None,
    ) -> "WaveFrame":
        """
        Trim the waveframe and return a copy.

        Unlike obspy's trim, this method only allows trimming the waveframe.
        If you would like to pad or extend the waveframes data use
        :func:`obsplus.WaveFrame.pad`.

        Parameters
        ----------
        {start_end_desc}

        Notes
        -----
        If a delta is used it should be positive for ``starttime`` and
        negative for ``endtime`` otherwise no trimming will be applied.
        """
        df = _Trimmer("trim")(self._df, starttime=starttime, endtime=endtime)
        return WaveFrame(df, processing=self.processing)

    @compose_docstring(start_end_desc=starttime_endtime_params)
    def cutout(
        self, starttime: trim_time_types, endtime: trim_time_types
    ) -> "WaveFrame":
        """
        Cut out sections of the WaveFrame.

        This method is essentially the opposite of
        :method:`obsplus.Waveframe.trim`.

        Parameters
        ----------
        {start_end_desc}
        """
        df = _Trimmer("cutout")(self._df, starttime=starttime, endtime=endtime)
        return WaveFrame(df, processing=self.processing)

    # --- Processing methods
    def detrend(self, method: str = "linear", **kwargs) -> "WaveFrame":
        """
        Detrend all traces in waveframe.

        Kwargs are passed to underlying functions.

        Parameters
        ----------
        method
            The detrend method to use. Supported options are:
                linear - Remove a linear trend for each row
                simple - Remove a lienar trend for first and last point of
                    each row
                constant - Subtract the mean of each row
        Notes
        -----
        When ``method == 'linear'`` detrending is performed either on each row,
        or if a row contains non-finite values (eg NaN Inf) detrending is
        performed on each contiguous segment. For example, consider the
        following data where two non-finite values exist in the first row
        has some NaN values (denoted by --):

            AAAAAAAAA--BBBBBBBBBB
            CCCCCCCCCCCCCCCCCCCCC

        In this case the detrending would be applied independently to segments
        marked with A, B, and C.
        """
        df = _WFDetrender(method)(self._df, **kwargs)
        return WaveFrame(df, processing=self.processing)


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
