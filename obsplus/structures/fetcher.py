"""
Data fetcher class stuff
"""
import copy
import functools
from collections import namedtuple
from functools import partial
from typing import Optional, Union, Callable, Tuple, Dict

import obspy
import pandas as pd
from obspy import Stream, UTCDateTime

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
    EVENT_WAVEFORM_PATH_STRUCTURE,
    WAVEFORM_STRUCTURE,
    get_waveforms_parameters,
    LARGEDT64,
)
from obsplus.events.utils import get_event_client
from obsplus.stations.utils import get_station_client
from obsplus.utils import (
    make_time_chunks,
    register_func,
    get_reference_time,
    filter_index,
    compose_docstring,
    to_datetime64,
    to_timedelta64,
    to_utc,
)
from obsplus.waveforms.utils import get_waveform_client

EventStream = namedtuple("EventStream", "event_id stream")


# ---------------------- fetcher constructor stuff


def _enable_swaps(cls):
    """ enable swapping out events, stations, and picks info on any
    function that gets waveforms """
    for name, value in cls.__dict__.items():
        if "waveform" in name or name == "__call__":
            setattr(cls, name, _temporary_override(value))
    return cls


def _temporary_override(func):
    """ decorator to enable temporary override of various parameters
    (eg events, stations, picks, etc.)"""

    @functools.wraps(func)
    def wraper(self, *args, **kwargs):
        inter = WAVEFETCHER_OVERRIDES.intersection(kwargs)
        if inter:
            self = self.copy()
            self.set_events(kwargs.pop("events", self.event_client))
            self.set_stations(kwargs.pop("stations", self.station_client))
            self.set_waveforms(kwargs.pop("waveforms", self.waveform_client))

        return func(self, *args, **kwargs)

    return wraper


# ---------------------------------- Wavefetcher class

fetcher_waveform_type = Union[waveform_clientable_type, obsplus.WaveBank]
fetcher_event_type = Union[event_clientable_type, pd.DataFrame, obsplus.EventBank]
fetcher_station_type = Union[station_clientable_type, pd.DataFrame]


@_enable_swaps
class Fetcher:
    """
    A class for serving up data from various sources.

    Integrates station, event, and waveform client for requests that
    are aware of the complete dataset.

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
    """

    # -------------------------------- class init

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
        self.time_before = time_before
        self.time_after = time_after
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
        self.waveform_df = None  # TODO figure out how to get waveform df?

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
        # a dict for filling missing values in station df
        try:
            self.station_client = get_station_client(stations)
        except TypeError:
            self.station_client = getattr(self, "station_client", None)
        try:
            self.station_df = stations_to_df(stations)
        except TypeError:
            # if unable to get station info from stations waveform client
            try:
                self.station_df = stations_to_df(self.waveform_client)
            except TypeError:
                #  if no waveforms try events
                try:
                    self.station_df = stations_to_df(self.event_client)
                except TypeError:
                    self.station_df = None

    # ------------------------ continuous data fetching methods

    @compose_docstring(get_waveforms_parameters=get_waveforms_parameters)
    def get_waveforms(self, *args, **kwargs) -> Stream:
        """
        Get waveforms for all channels in stations.

        {get_waveform_parameters}

        Returns
        -------
        obspy.Stream
        """
        # get a list of tuples for get_waveforms_bulk call
        nslcs = self._get_bulk_arg(*args, **kwargs)
        return self._get_bulk_wf(nslcs)

    def yield_waveforms(
        self,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        duration: float,
        overlap: float,
    ) -> Stream:
        """
        Yield streams from starttime to endtime of a specified duration.

        Parameters
        ----------
        starttime
            The starting time of the streams
        endtime
            The stopping time of the streams
        duration
            The duration of each waveforms between starttime and endtime
        overlap
            The overlap between streams added to the end of the waveforms

        Yields
        -------
        Stream
        """
        time_chunks = make_time_chunks(starttime, endtime, duration, overlap)
        for t1, t2 in time_chunks:
            yield self.get_waveforms(starttime=t1, endtime=t2)

    def yield_waveform_callable(
        self,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        duration: float,
        overlap: float,
    ) -> Callable[..., Stream]:
        """
        Yield callables that take no arguments and return a waveforms.

        This function is useful if you are distributing work to remote
        processes and want to avoid serializing the waveforms.

        Parameters
        ----------
        starttime
            The starting time of the streams
        endtime
            The stopping time of the streams
        duration
            The duration of each waveforms between starttime and endtime
        overlap
            The overlap between streams added to the end of the waveforms

        Yields
        -------
        Callable[..., Stream]
        """

        time_chunks = make_time_chunks(starttime, endtime, duration, overlap)
        for t1, t2 in time_chunks:

            def _func(starttime=t1, endtime=t2):
                return self.get_waveforms(starttime=starttime, endtime=endtime)

            yield _func

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
        Yield event_id and streams for each event in the events.

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
        assert reference.lower() in self.reference_funcs
        tb = to_timedelta64(time_before, default=self.time_before)
        ta = to_timedelta64(time_after, default=self.time_after)
        assert (tb is not None) and (ta is not None)
        # get reference times
        event_ids = self.event_df.event_id.values
        reftimes = {x: self.reference_funcs[reference](self, x) for x in event_ids}
        # if using a wavebank preload index over entire time-span for speedup
        if isinstance(self.waveform_client, WaveBank):
            mt = min([x.min() if hasattr(x, "min") else x for x in reftimes.values()])
            mx = max([x.max() if hasattr(x, "max") else x for x in reftimes.values()])
            index = self.waveform_client.read_index(starttime=mt, endtime=mx)
            get_bulk_wf = partial(self._get_bulk_wf, index=index)
        else:
            get_bulk_wf = self._get_bulk_wf
        # iterate each event in the events and yield the waveform
        for event_id in event_ids:
            # make sure ser is either a single datetime or a series of datetimes
            ti_ = to_datetime64(reftimes[event_id])
            bulk_args = self._get_bulk_arg(starttime=ti_ - tb, endtime=ti_ + ta)
            try:
                yield EventStream(event_id, get_bulk_wf(bulk_args))
            except Exception:
                if raise_on_fail:
                    raise

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

    def copy(self):
        return copy.deepcopy(self)

    def _get_bulk_wf(self, *args, **kwargs):
        """
        get the wave forms using the client, apply processor if it is defined
        """
        out = self.waveform_client.get_waveforms_bulk(*args, **kwargs)
        if out is None:
            out = self.waveform_client.get_waveforms_bulk(*args, **kwargs)

        if callable(self.stream_processor):
            return self.stream_processor(out) or out
        else:
            return out

    def _get_bulk_arg(self, starttime=None, endtime=None, **kwargs) -> list:
        """ get the argument passed to get_waveforms_bulk, see
        obspy.fdsn.client for more info """
        station_df = self.station_df.copy()
        inv = station_df[filter_index(station_df, **kwargs)]
        # replace None/Nan with larger number
        inv.loc[inv["end_date"].isnull(), "end_date"] = LARGEDT64
        inv["end_date"] = inv["end_date"].astype("datetime64[ns]")
        # remove station/channels that dont have data for requested time
        starttime = to_datetime64(starttime, default=inv["start_date"].min())
        endtime = to_datetime64(endtime, default=inv["end_date"].max())
        con1, con2 = (inv["start_date"] > endtime), (inv["end_date"] < starttime)

        inv = inv[~(con1 | con2)]
        df = inv[list(NSLC)]
        if df.empty:  # return empty list if no data found
            return []
        df.loc[:, "starttime"] = starttime
        df.loc[:, "endtime"] = endtime
        # remove any rows that don't have defined start/end times
        out = df[(~df["starttime"].isnull()) & (~df["endtime"].isnull())]
        # ensure we have UTCDateTime objects
        out["starttime"] = [to_utc(x) for x in df["starttime"]]
        out["endtime"] = [to_utc(x) for x in df["endtime"]]
        # convert to list of tuples and return
        return [tuple(x) for x in out.to_records(index=False)]

    def download_waveforms(
        self,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        duration: float,
        overlap: float = 0,
        path: str = WAVEFORM_STRUCTURE,
    ) -> None:
        """
        Download contiguous waveform data and save in directory.

        Parameters
        ----------
        starttime
            The start time of the data  to download
        endtime
            The end time of the data to download
        duration
            The duration of each chunk of data in seconds
        overlap
            The overlap, added to the end of each waveforms, for the data
        path
            A string that specifies the directory structure. See the
            path_structure argument of :class: `~obsplus.Sbank` for more info
        """
        bank = WaveBank(path)
        # iter events and save to disk
        t1, t2 = starttime, endtime
        for stream in self.yield_waveforms(t1, t2, duration, overlap):
            bank.put_waveforms(stream)

    def download_event_waveforms(
        self,
        time_before_origin: float,
        time_after_origin: float,
        path: str = EVENT_WAVEFORM_PATH_STRUCTURE,
    ) -> None:
        """
        Download waveforms corresponding to events in waveforms from client.

        Parameters
        ----------
        time_before_origin
            The number of seconds before the reported origin to include in the
            event waveforms
        time_after_origin
            The number of seconds after the reported origin to include in the
            event waveforms
        path
            A string that specifies the directory structure. See the
            path_structure argument of :class: `~obsplus.Sbank` for more info.
        """
        # setup banks and fetcher
        bank = WaveBank(path)
        # iter events and save to disk
        t1, t2 = time_before_origin, time_after_origin
        for event_id, stream in self.yield_event_waveforms(t1, t2):
            bank.put_waveforms(stream, name=event_id)

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
def _get_origin_reference_times(fetcher: Fetcher, event_id):
    """ get the reference times for """
    df = fetcher.event_df
    row = df[df.event_id == event_id].iloc[0]
    return UTCDateTime(row.time)


@register_func(Fetcher.reference_funcs, key="p")
def _get_p_reference_times(fetcher: Fetcher, event_id):
    """ get the reference times for """
    return _get_phase_reference_time(fetcher, event_id, "p")


@register_func(Fetcher.reference_funcs, key="s")
def _get_s_reference_times(fetcher: Fetcher, event_id):
    """ get the reference times for """
    return _get_phase_reference_time(fetcher, event_id, "s")


def _get_phase_reference_time(fetcher: Fetcher, event_id, phase):
    """ get reference times to specified phases, apply over all channels in a
    station """
    pha = phase.upper()
    df = fetcher.picks_df
    inv = fetcher.station_df
    assert df is not None and inv is not None
    assert (df.phase_hint.str.upper() == pha).any(), f"no {phase} picks found"
    dff = df[(df.event_id == event_id) & (df.phase_hint == pha)]
    merge = pd.merge(inv, dff[["time", "station"]], on="station", how="left")
    return merge["time"]
