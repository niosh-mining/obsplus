"""
Data fetcher class stuff
"""
import copy
import functools
import warnings
from collections import namedtuple
from functools import partial
from typing import Optional, Union, Callable, Tuple, Dict

import numpy as np
import obspy
import pandas as pd
from obspy import Stream

import obsplus
from obsplus import events_to_df, stations_to_df, picks_to_df
from obsplus.bank.wavebank import WaveBank
from obsplus.constants import (
    waveform_clientable_type,
    event_clientable_type,
    station_clientable_type,
    stream_proc_type,
    NSLC,
    WAVEFETCHER_OVERRIDES,
    event_time_type,
    get_waveforms_parameters,
    bulk_waveform_arg_type,
    LARGEDT64,
)
from obsplus.exceptions import TimeOverflowWarning
from obsplus.utils.docs import compose_docstring
from obsplus.utils.events import get_event_client
from obsplus.utils.misc import register_func, suppress_warnings
from obsplus.utils.pd import filter_index, get_seed_id_series
from obsplus.utils.stations import get_station_client
from obsplus.utils.time import get_reference_time
from obsplus.utils.time import to_datetime64, to_timedelta64, make_time_chunks, to_utc
from obsplus.utils.waveforms import get_waveform_client

EventStream = namedtuple("EventStream", "event_id stream")


# ---------------------- fetcher constructor stuff


def _enable_swaps(cls):
    """
    Enable swapping out events, stations, and picks info on any
    function that gets waveforms.
    """
    for name, value in cls.__dict__.items():
        if "waveform" in name or name == "__call__":
            setattr(cls, name, _temporary_override(value))
    return cls


def _temporary_override(func):
    """
    Decorator to enable temporary override of various parameters.

    This is commonly used for events, stations, picks, etc.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        inter = WAVEFETCHER_OVERRIDES.intersection(kwargs)
        if inter:
            self = self.copy()
            self.set_events(kwargs.pop("events", self.event_client))
            self.set_stations(kwargs.pop("stations", self.station_client))
            self.set_waveforms(kwargs.pop("waveforms", self.waveform_client))

        return func(self, *args, **kwargs)

    return wrapper


# ---------------------------------- Wavefetcher class

fetcher_waveform_type = Union[waveform_clientable_type, obsplus.WaveBank]
fetcher_event_type = Union[event_clientable_type, pd.DataFrame, obsplus.EventBank]
fetcher_station_type = Union[station_clientable_type, pd.DataFrame]


@_enable_swaps
class Fetcher:
    """
    A class for serving up data from various sources.

    Integrates station, event, and waveform clients to enable dataset-aware
    querying.

    Parameters
    ----------
    waveforms
        Any argument from which a waveform client can be extracted. This
        includes an obspy waveforms, directory of waveform files, or an object
        with a `get_waveforms` method.
    stations
        Any argument from which an station client can be extracted. This
        includes an obspy Inventory, directory of station files, or an object
        with a `get_stations` method.
    events
        Any argument from which an event client can be extracted. This
        includes an obspy Catalog, directory of event files, or an object with
        a `get_events` method.
    picks
        Data from which picks can be extracted. A dataframe, events, or
        event_client are all acceptable.
    stream_processor
        A callable that takes an obspy waveforms as input and returns an obspy
        waveforms.
    time_before
        The default time before an given time to fetch.
    time_after
        The default time after a supplied time to fetch.
    event_query
        A dict of arguments used to filter events.

    Examples
    --------
    >>> import obsplus
    >>> import obspy

    >>> #--- Init a Fetcher
    >>> # from a dataset
    >>> ds = obsplus.load_dataset('bingham_test')
    >>> ds_fetcher = ds.get_fetcher()
    >>> assert isinstance(ds_fetcher, obsplus.Fetcher)
    >>> # from separate clients (includes Stream, Inventory, Catalog)
    >>> waveforms = ds.waveform_client
    >>> events = ds.event_client
    >>> stations = ds.station_client
    >>> kwargs = dict(events=events, waveforms=waveforms, stations=stations)
    >>> fetcher = obsplus.Fetcher(**kwargs)
    >>> assert isinstance(fetcher, obsplus.Fetcher)

    >>> # --- get contiguous (not event) waveform data
    >>> # simple get_waveform calls are passed to the waveforms client
    >>> fetcher = obsplus.load_dataset('ta_test').get_fetcher()
    >>> t1 = obspy.UTCDateTime('2007-02-15')
    >>> t2 = t1 + 60
    >>> station = 'M14A'
    >>> st = fetcher.get_waveforms(starttime=t1, endtime=t2, station=station)
    >>> print(st)
    3 Trace(s) ...
    >>> # iterate over a range of times
    >>> t1 = obspy.UTCDateTime('2007-02-16')
    >>> t2 = t1 + (3600 * 24)
    >>> for st in fetcher.yield_waveforms(starttime=t1, endtime=t2):
    ...     assert len(st)

    >>> # --- get event waveforms
    >>> fetcher = obsplus.load_dataset('bingham_test').get_fetcher()
    >>> # iterate each event yielding streams 30 seconds after origin
    >>> kwargs = dict(time_before=0, time_after=30, reference='origin')
    >>> for event_id, st in fetcher.yield_event_waveforms(**kwargs):
    ...     assert isinstance(event_id, str)
    ...     assert isinstance(st, obspy.Stream)
    """

    def __init__(
        self,
        waveforms: fetcher_waveform_type,
        stations: Optional[fetcher_event_type] = None,
        events: Optional[fetcher_event_type] = None,
        picks: Optional[pd.DataFrame] = None,
        stream_processor: Optional[stream_proc_type] = None,
        time_before: Optional[float] = None,
        time_after: Optional[float] = None,
        event_query: Optional[dict] = None,
    ):
        # if fetch_arg is a WaveFetcher just update dict and return
        if isinstance(waveforms, Fetcher):
            self.__dict__.update(waveforms.__dict__)
            return
        # get clients for each data types
        self.set_waveforms(waveforms)
        self.set_events(events)
        self.set_stations(stations)
        self._picks_input = picks
        # waveforms processor for applying filters and such
        self.stream_processor = stream_processor
        # set event time/query parameters
        self.time_before = to_timedelta64(time_before)
        self.time_after = to_timedelta64(time_after)
        self.event_query = event_query or {}

    def set_waveforms(self, waveforms: fetcher_waveform_type):
        """
        Set waveform state in Fetcher.

        Parameters
        ----------
        waveforms
            Data representing waveforms, from which a wave client can be
            inferred.
        """
        try:
            self.waveform_client = get_waveform_client(waveforms)
        except TypeError:  # if the waveform client is already defined keep it
            self.waveform_client = getattr(self, "waveform_client", None)

    def set_events(self, events: fetcher_event_type):
        """
        Set event state in fetcher.

        Parameters
        ----------
        events
            Data representing events, from which a client or dataframe can
            be obtained.
        """
        # set event and dataframe
        try:
            self.event_client = get_event_client(events)
        except TypeError:
            self.event_client = getattr(self, "event_client", None)
        try:
            self.event_df = events_to_df(events)
        except TypeError:
            self.event_df = None
        self._picks_df = None

    def set_stations(self, stations: fetcher_station_type):
        """
        Set the station state in fetcher.

        Parameters
        ----------
        stations
            Data representing stations, from which a client or dataframe
            can be inferred.
        """
        try:
            self.station_client = get_station_client(stations)
        except TypeError:
            self.station_client = getattr(self, "station_client", None)
        try:
            # since its common for inventories to have far out enddates this
            # can raise a warning. These are safe to ignore.
            with suppress_warnings(category=TimeOverflowWarning):
                self.station_df = stations_to_df(stations)
        except TypeError:
            # if unable to get station info from stations use waveform client
            try:
                self.station_df = stations_to_df(self.waveform_client)
            except TypeError:
                #  if no waveforms try events
                try:
                    self.station_df = stations_to_df(self.event_client)
                except TypeError:
                    self.station_df = None
        # make sure seed_id is set
        if self.station_df is not None:
            self.station_df["seed_id"] = get_seed_id_series(self.station_df)

    # ------------------------ continuous data fetching methods

    @compose_docstring(get_waveforms_parameters=get_waveforms_parameters)
    def get_waveforms(self, *args, **kwargs) -> Stream:
        """
        Get waveforms for all channels in stations.

        {get_waveform_parameters}
        """
        # get a list of tuples for get_waveforms_bulk call
        nslcs = self._get_bulk_args(*args, **kwargs)
        return self._get_bulk_wf(nslcs)

    @compose_docstring(get_waveforms_params=get_waveforms_parameters)
    def yield_waveforms(
        self,
        network: Optional[str] = None,
        station: Optional[str] = None,
        location: Optional[str] = None,
        channel: Optional[str] = None,
        starttime: Optional[obspy.UTCDateTime] = None,
        endtime: Optional[obspy.UTCDateTime] = None,
        duration: float = 3600.0,
        overlap: Optional[float] = None,
    ) -> Stream:
        """
        Yield time-series segments from the waveform client.

        Parameters
        ----------
        {get_waveforms_params}
        duration : float
            The duration of the streams to yield. All channels selected
            channels will be included in the waveforms.
        overlap : float
            If duration is used, the amount of overlap in yielded streams,
            added to the end of the waveforms.

        Notes
        -----
        All string parameters can use posix style matching with * and ? chars.

        Total duration of yielded streams = duration + overlap.

        If no starttime or endtime is provided the min/max indicated by the
        stations will be used.
        """
        # Note: although WaveBank has a yield waveforms method, we want
        # fetcher to work with any client so we don't use its implementation.
        starttime = to_utc(starttime or self.station_df["start_date"].min())
        endtime = to_utc(endtime or self.station_df["end_date"].max())
        time_chunks = make_time_chunks(starttime, endtime, duration, overlap)
        for t1, t2 in time_chunks:
            kwargs = dict(
                network=network, station=station, location=location, channel=channel
            )
            yield self.get_waveforms(starttime=t1, endtime=t2, **kwargs)

    # ------------------------ event waveforms fetching methods

    reference_funcs = {}  # stores funcs for getting event reference times

    def yield_event_waveforms(
        self,
        time_before: Optional[float] = None,
        time_after: Optional[float] = None,
        reference: Union[str, Callable] = "origin",
        raise_on_fail: bool = True,
    ) -> Tuple[str, Stream]:
        """
        Yield event_id and streams for each event.

        Parameters
        ----------
        time_before
            The time before (in seconds) the reference that will be included
            in the waveforms if possible.
        time_after
            The Time after (in seconds) the reference that will be included
            in the waveforms if possible.
        reference
            A str that indicates how the starttime of the trace should be
            determined. The following are supported:
                origin - use the origin time of the event
                p - use the first p time as the start for each station
                s - use the first s times as the start for each station
            If "p" or "s" is used only streams corresponding to stations with
            the appropriate phase pick will be returned.
        raise_on_fail
            If True, re raise an exception if one is caught during waveform
            fetching, else continue to next event.

        Notes
        -----
        Streams will not be yielded for any event for which a reference time
        cannot be obtained. For example, if reference='S' only events with some
        S picks will be yielded.
        """

        def _check_yield_event_waveform_(reference, ta, tb):
            if not reference.lower() in self.reference_funcs:
                msg = (
                    f"reference of {reference} is not supported. Supported "
                    f"reference arguments are {list(self.reference_funcs)}"
                )
                raise ValueError(msg)
            if not (np.abs(tb) + np.abs(ta)) > np.timedelta64(0, "s"):
                msg = (
                    "time_before and/or time_after must be specified in either "
                    "Fetcher's init or the yield_event_Waveforms call"
                )
                raise ValueError(msg)

        tb = to_timedelta64(time_before, default=self.time_before)
        ta = to_timedelta64(time_after, default=self.time_after)
        _check_yield_event_waveform_(reference, ta, tb)
        # get reference times
        ref_func = self.reference_funcs[reference.lower()]
        reftime_df = ref_func(self)
        # if using a wavebank preload index over entire time-span for speedup
        if isinstance(self.waveform_client, WaveBank) and len(reftime_df):
            mt, mx = reftime_df["time"].min(), reftime_df["time"].max()
            index = self.waveform_client.read_index(starttime=mt, endtime=mx)
            get_bulk_wf = partial(self._get_bulk_wf, index=index)
        else:
            get_bulk_wf = self._get_bulk_wf
        # iterate each event in the events and yield the waveform
        for event_id, df in reftime_df.groupby("event_id"):
            # make sure ser is either a single datetime or a series of datetimes
            time = to_datetime64(df["time"])
            t1, t2 = time - tb, time + ta
            bulk_args = self._get_bulk_args(starttime=t1, endtime=t2)
            try:
                yield EventStream(event_id, get_bulk_wf(bulk_args))
            except Exception:
                if raise_on_fail:
                    raise
                else:
                    msg = f"Fetcher failed to get waveforms for {event_id}."
                    warnings.warn(msg)

    def get_event_waveforms(
        self,
        time_before: Optional[float] = None,
        time_after: Optional[float] = None,
        reference: Union[str, Callable] = "origin",
        raise_on_fail: bool = True,
    ) -> Dict[str, Stream]:
        """
        Return a dict of event_ids and waveforms for each event in events.

        Parameters
        ----------
        time_before
            The time before (in seconds) the reference that will be included
            in the waveforms if possible.
        time_after
            The Time after (in seconds) the reference that will be included
            in the waveforms if possible.
        reference
            A str that indicates how the starttime of the trace should be
            determined. The following are supported:
                origin - use the origin time of the event for each channel
                p - use the first p times as the start of the station traces
                s - use the first s times as the start of the station traces
            If a station doesn't have p or s picks and "p" or "s" is used,
            it's streams will not be returned.
        raise_on_fail
            If True, re raise and exception if one is caught during waveform
            fetching, else continue to next event.

        Yields
        ------
        obspy.Stream
        """
        inputs = dict(
            time_before=time_before,
            time_after=time_after,
            reference=reference,
            raise_on_fail=raise_on_fail,
        )
        return dict(self.yield_event_waveforms(**inputs))

    def __call__(
        self,
        time_arg: event_time_type,
        time_before: Optional[float] = None,
        time_after: Optional[float] = None,
        *args,
        **kwargs,
    ) -> obspy.Stream:
        """
        Using a reference time, return a waveforms that encompasses that time.

        Parameters
        ----------
        time_arg
            The argument that will indicate a start time. Can be a one
            length events, and event, a float, or a UTCDatetime object
        time_before
            The time before time_arg to include in waveforms
        time_after
            The time after time_arg to include in waveforms

        Returns
        -------
        obspy.Stream
        """
        tbefore = to_timedelta64(time_before, default=self.time_before)
        tafter = to_timedelta64(time_after, default=self.time_after)
        assert (tbefore is not None) and (tafter is not None)
        # get the reference time from the object
        time = to_datetime64(get_reference_time(time_arg))
        t1 = time - tbefore
        t2 = time + tafter
        return self.get_waveforms(starttime=to_utc(t1), endtime=to_utc(t2), **kwargs)

    # ------------------------------- misc

    def copy(self) -> "Fetcher":
        """Return a deep copy of the fetcher."""
        return copy.deepcopy(self)

    def _get_bulk_wf(self, *args, **kwargs):
        """
        get the wave forms using the client, apply processor if it is defined
        """
        out = self.waveform_client.get_waveforms_bulk(*args, **kwargs)
        if callable(self.stream_processor):
            return self.stream_processor(out) or out
        else:
            return out

    def _get_bulk_args(
        self, starttime=None, endtime=None, **kwargs
    ) -> bulk_waveform_arg_type:
        """
        Get the bulk waveform arguments based on given start/end times.

        This method also takes into account data availability as contained
        in the stations data.

        Parameters
        ----------
        starttime
            Start times for query.
        endtime
            End times for query.

        Returns
        -------
        List of tuples of the form:
            [(network, station, location, channel, starttime, endtime)]
        """
        station_df = self.station_df.copy()
        inv = station_df[filter_index(station_df, **kwargs)]
        # replace None/Nan with larger number
        inv.loc[inv["end_date"].isnull(), "end_date"] = LARGEDT64
        inv["end_date"] = inv["end_date"].astype("datetime64[ns]")
        # get start/end of the inventory
        inv_start = inv["start_date"].min()
        inv_end = inv["end_date"].max()
        # remove station/channels that dont have data for requested time
        min_time = to_datetime64(starttime, default=inv_start).min()
        max_time = to_datetime64(endtime, default=inv_end).max()
        con1, con2 = (inv["start_date"] > max_time), (inv["end_date"] < min_time)
        df = inv[~(con1 | con2)].set_index("seed_id")[list(NSLC)]
        if df.empty:  # return empty list if no data found
            return []
        if isinstance(starttime, pd.Series):
            # Have to get clever here to make sure only active stations get used
            # and indices are not duplicated.
            new_start = starttime.loc[set(starttime.index).intersection(df.index)]
            new_end = endtime.loc[set(endtime.index).intersection(df.index)]
            df["starttime"] = new_start.loc[~new_start.index.duplicated()]
            df["endtime"] = new_end.loc[~new_end.index.duplicated()]
        else:
            df["starttime"] = starttime
            df["endtime"] = endtime
        # remove any rows that don't have defined start/end times
        out = df[~(df["starttime"].isnull() | df["endtime"].isnull())]
        # ensure we have UTCDateTime objects
        out["starttime"] = [to_utc(x) for x in out["starttime"]]
        out["endtime"] = [to_utc(x) for x in out["endtime"]]
        # convert to list of tuples and return
        return [tuple(x) for x in out.to_records(index=False)]

    @property
    def picks_df(self):
        """ return a dataframe from the picks (if possible) """
        if self._picks_df is None:
            try:
                df = picks_to_df(self.event_client)
            except TypeError:
                self._picks_df = None
            else:
                self._picks_df = df
        return self._picks_df

    @picks_df.setter
    def picks_df(self, item):
        setattr(self, "_picks_df", item)


# ------------------------ functions for getting reference times


@register_func(Fetcher.reference_funcs, key="origin")
def _get_origin_reference_times(fetcher: Fetcher) -> pd.Series:
    """Get the reference times for origins."""
    event_df = fetcher.event_df[["time", "event_id"]].set_index("event_id")
    inv_df = fetcher.station_df
    # iterate each event and add rows for each channel in inventory
    dfs = []
    for eid, ser in event_df.iterrows():
        inv = inv_df.copy()
        inv["event_id"] = eid
        inv["time"] = ser["time"]
        dfs.append(inv)
    # get output
    out = (
        pd.concat(dfs, ignore_index=True)
        .reset_index()
        .set_index("seed_id")[["event_id", "time"]]
        .dropna(subset=["time"])
    )
    return out


@register_func(Fetcher.reference_funcs, key="p")
def _get_p_reference_times(fetcher: Fetcher) -> pd.Series:
    """Get the reference times for p arrivals."""
    return _get_phase_reference_time(fetcher, "p")


@register_func(Fetcher.reference_funcs, key="s")
def _get_s_reference_times(fetcher: Fetcher) -> pd.Series:
    """Get the reference times for s arrivals."""
    return _get_phase_reference_time(fetcher, "s")


def _get_phase_reference_time(fetcher: Fetcher, phase):
    """
    Get reference times to specified phases, apply over all channels in a
    station.
    """
    pha = phase.upper()
    # ensure the pick_df and inventory df exist
    pick_df = fetcher.picks_df
    inv_df = fetcher.station_df
    assert pick_df is not None and inv_df is not None
    # filter dataframes for phase of interest
    assert (pick_df["phase_hint"].str.upper() == pha).any(), f"no {phase} picks found"
    pick_df = pick_df[pick_df["phase_hint"] == pha]
    # merge inventory and pick df together, ensure time is datetime64
    columns = ["time", "station", "event_id"]
    merge = pd.merge(inv_df, pick_df[columns], on="station", how="left")
    merge["time"] = to_datetime64(merge["time"])
    assert merge["seed_id"].astype(bool).all()
    return merge.set_index("seed_id")[["time", "event_id"]]
