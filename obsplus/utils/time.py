"""Obsplus utilities for working with times."""
import warnings
from functools import singledispatch
from typing import Union, Optional, Sequence, Dict, Any, TypeVar, Generator, Tuple

import numpy as np
import obspy
import pandas as pd
from obspy.core import event as ev

from obsplus.constants import (
    event_time_type,
    wave_type,
    utc_able_type,
    DEFAULT_TIME,
    LARGEDT64,
    SMALLDT64,
    TIME_COLUMNS,
    relative_time_types,
)
from obsplus.exceptions import TimeOverflowWarning
from obsplus.utils.docs import compose_docstring

rtype = relative_time_types


@singledispatch
def get_reference_time(
    obj: Optional[Union[event_time_type, wave_type]],
) -> Optional[obspy.UTCDateTime]:
    """
    Get a reference time inferred from an object.

    Parameters
    ----------
    obj
        The argument that will indicate a start time. Types and corresponding
        behavior are as follows:
            float:
                Convert to UTCDateTime object, interpret as a timestamp.
            UTCDateTime:
                Return a new UTCDateTime as a copy.
            catalog:
                Return the earliest reference time of all events.
            event:
                First check for a preferred origin, if found return its origin
                time. If not found, iterate through the events picks and
                return the earliest pick time. If None are found raise a
                ValueError.
            stream:
                Return the earliest reference time of all traces.
            trace:
                Return the starttime in the stats object.

    Raises
    ------
    TypeError if the type is not supported.
    ValueError if the type is supported but no reference time could be determined.

    Returns
    -------
    obspy.UTCDateTime
    """
    if obj is None:
        return None
    try:
        return to_utc(obj)
    except TypeError:
        msg = f"get_reference_time does not support type {type(obj)}"
        raise TypeError(msg)


@get_reference_time.register(ev.Event)
def _get_event_origin_time(event):
    """ get the time from preferred origin from the event """
    # try to get origin
    try:
        por = event.preferred_origin() or event.origins[-1]
    except IndexError:
        por = None
    if por is not None:
        if por.time is None:
            msg = f"no origin time found on {por}"
            raise ValueError(msg)
        return get_reference_time(por.time)
    # else try using picks
    elif event.picks:
        return get_reference_time(event.picks)
    else:
        msg = f"could not get reference time for {event}"
        raise ValueError(msg)


@get_reference_time.register(ev.Origin)
def _get_origin_time(origin):
    return get_reference_time(origin.time)


@get_reference_time.register(ev.Pick)
def _get_first_pick(pick):
    """Get the reference time from a single pick."""
    return get_reference_time(pick.time)


@get_reference_time.register(list)
def _from_list(input_list):
    """Get reference time as minimum in a list."""
    outs = [get_reference_time(x) for x in input_list]
    return min([x for x in outs if x is not None])


@get_reference_time.register(obspy.Catalog)
def _get_first_event(catalog):
    """ ensure the events is length one, return event """
    assert len(catalog) == 1, f"{catalog} has more than one event"
    return _get_event_origin_time(catalog[0])


@get_reference_time.register(obspy.Stream)
def _get_stream_time(st):
    """ return the earliest start time for stream. """
    return min([get_reference_time(tr) for tr in st])


@get_reference_time.register(obspy.Trace)
def _get_trace_time(tr):
    """ return starttime of trace. """
    return tr.stats.starttime


@singledispatch
def to_datetime64(
    value: Optional[Union[utc_able_type, Sequence[utc_able_type]]], default=DEFAULT_TIME
) -> Union[np.datetime64, np.ndarray, pd.Series]:
    """
    Convert time value to a numpy datetime64, or array of such.

    This function essentially uses the same logic as ~:class:`obspy.UTCDateTime`
    but also accounts for native numpy types.

    Parameters
    ----------
    value
        Any value that can be interpreted as a time. If a sequence is passed
        an ndarray of type "datetime64[ns]" is returned.
    default
        A value for missing data. pandas.NaT is used by default.

    Examples
    --------
    >>> import obspy
    >>> import pandas as pd
    >>> timestr = '2020-01-03T11:00:00'
    >>> inputs = [obspy.UTCDateTime(timestr), None, 2525, np.datetime64(timestr)]
    >>> out = to_datetime64(inputs)
    >>> for t in out:
    ...     assert isinstance(t, np.datetime64) or pd.isnull(t)
    >>> assert pd.isnull(out[1])

    """
    # null values return default (usually NaT)
    if pd.isnull(value):
        if not pd.isnull(default):
            return to_datetime64(default)
        return default
    elif isinstance(value, np.datetime64):
        # The '.astype('datetime64[ns]') is necessary to make sure it really
        # does get converted to nanoseconds
        return value.astype("datetime64[ns]")
    elif isinstance(value, pd.Timestamp):
        # The '.astype('datetime64[ns]') is necessary to make sure it really
        # does get converted to nanoseconds
        return value.to_datetime64().astype("datetime64[ns]")
    try:
        utc = obspy.UTCDateTime(value)
        return np.datetime64(utc._ns, "ns")
    # the UTCDateTime is too big or small, use biggest/smallest values instead
    except (SystemError, OverflowError):
        new = LARGEDT64 if np.sign(utc._ns) > 0 else SMALLDT64
        msg = (
            f"time is too large to represent with a int64 with ns precision,"
            f" downgrading to {new}"
        )
        warnings.warn(msg, category=TimeOverflowWarning)
        return new


@to_datetime64.register(str)
def _from_string(time_str: str, default=DEFAULT_TIME):
    """ Convert to a string. """
    if not time_str:
        return default
    return to_datetime64(obspy.UTCDateTime(time_str))


@to_datetime64.register(pd.Series)
def _series_to_datetime(value, default=DEFAULT_TIME):
    """ Convert a series to datetimes """
    return value.apply(to_datetime64, default=default)


@to_datetime64.register(np.ndarray)
def _ndarray_to_datetime64(value, default=DEFAULT_TIME):
    """ Convert an array to datetimes. """
    ns = np.array([to_datetime64(x, default=default) for x in value])
    return pd.to_datetime(ns, unit="ns").values


@to_datetime64.register(list)
@to_datetime64.register(tuple)
def _sequence_to_datetime64(value, default=DEFAULT_TIME):
    out = [to_datetime64(x, default=default) for x in value]
    return np.array(out)


def to_utc(
    value: Union[utc_able_type, Sequence[utc_able_type]]
) -> Union[obspy.UTCDateTime, np.ndarray]:
    """
    Convert an object or sequence to a UTCDateTime object.

    Parameters
    ----------
    value
        Any value readable by ~:class:`obspy.UTCDateTime`,
        ~:class:`numpy.datetime64` or a sequence of such.

    Examples
    --------
    >>> import numpy as np
    >>> import obspy
    >>> inputs = [1, '2017-09-18', np.datetime64('2020-01-03')]
    >>> out = to_utc(inputs)
    >>> for utc in out:
    ...     assert isinstance(utc, obspy.UTCDateTime)
    """

    def _dt64_to_utc(dt64):
        if isinstance(dt64, pd.Timestamp):
            dt64 = dt64.to_datetime64()
        ns = dt64.astype("datetime64[ns]").astype(np.int64)
        return obspy.UTCDateTime(ns=int(ns))

    # just use to_datetime64 for flexible handling of types
    dt64ish = to_datetime64(value)
    # return
    if isinstance(dt64ish, np.datetime64):
        return _dt64_to_utc(dt64ish)
    # else assume a sequence of some sort and every element is a dt64
    seq = [_dt64_to_utc(x) for x in dt64ish]
    return np.array(seq)


@singledispatch
def to_timedelta64(
    value: Optional[Union[rtype, Sequence[rtype], np.ndarray, pd.Series]],
    default: Any = np.timedelta64(0, "s"),
) -> Union[np.timedelta64, np.ndarray, pd.Series]:
    """
    Convert a value to a timedelta[ns].

    Numpy does not gracefully handle non-ints so we need to do some rounding
    first.

    Parameters
    ----------
    value
        A float or an int to convert to datetime.
    default
        The default to return if the input value is not truthy.

    Examples
    --------
    >>> import numpy as np
    >>> # get array of timedelta64[ns] from various input types
    >>> inputs = [1, 2.3232, np.timedelta64(5, 'Y')]
    >>> out = to_timedelta64(inputs)
    >>> assert isinstance(out, np.ndarray)
    """
    if pd.isnull(value):
        return np.timedelta64(default)
    if isinstance(value, np.timedelta64):
        return value
    if isinstance(value, pd.Timedelta):
        return value.to_timedelta64()
    ns = to_utc(value)._ns
    return np.timedelta64(ns, "ns")


@to_timedelta64.register(tuple)
@to_timedelta64.register(list)
def _list_tuple_to_timedelta(value, default=None):
    """ Convert sequences to timedeltas. """
    out = [to_datetime64(x, default=default) for x in value]
    return np.array(out)


@to_timedelta64.register(pd.Series)
def _series_to_timedelta(ser) -> pd.Series:
    """ Convert a series to a timedelta. """
    return ser.apply(to_timedelta64)


@to_timedelta64.register(np.ndarray)
def _array_to_timedelta(obj):
    """ Convert a series to a timedelta. """
    if np.issubdtype(obj.dtype, np.timedelta64):
        return obj
    return np.array([to_timedelta64(x) for x in obj])


@compose_docstring(time_keys=str(TIME_COLUMNS))
def _dict_times_to_npdatetimes(
    input_dict: Dict[str, Any], time_keys: Sequence[str] = TIME_COLUMNS
) -> Dict[str, Any]:
    """
    Ensure time values in input_dict are converted to np.datetime64.

    Parameters
    ----------
    input_dict
        A dict that may contain time representations.
    time_keys
        A sequence of keys to search for and convert to np.datetime64.
        Defaults are:
        {time_keys}
    """
    out = dict(input_dict)
    for time_key in set(input_dict) & set(time_keys):
        out[time_key] = to_datetime64(out[time_key])
    return out


@compose_docstring(time_keys=str(TIME_COLUMNS))
def _dict_times_to_ns(
    input_dict: Dict[str, Any], time_keys: Sequence[str] = TIME_COLUMNS
) -> Dict[str, Any]:
    """
    Ensure time values in input_dict are converted to ints (ns).

    Parameters
    ----------
    input_dict
        A dict that may contain time representations.
    time_keys
        A sequence of keys to search for and convert to np.datetime64.
        Defaults are:
        {time_keys}
    """
    out = dict(input_dict)
    for time_key in set(input_dict) & set(time_keys):
        if not isinstance(out[time_key], int):  # assume ints are ns
            out[time_key] = to_datetime64(out[time_key]).astype(np.int64)
    return out


utc_var = TypeVar("utc_var", utc_able_type, Sequence[utc_able_type])


def make_time_chunks(
    utc1: utc_able_type,
    utc2: utc_able_type,
    duration: Union[float, int],
    overlap: Union[float, int] = 0.0,
) -> Generator[Tuple[obspy.UTCDateTime, obspy.UTCDateTime], None, None]:
    """
    Yield time intervals fitting in given datetime range.

    Function takes two utc date time objects and create a generator to
    yield all time in between by intervals of duration. Overlap is number
    of seconds segment n will overlap into segment n + 1.

    Parameters
    ----------
    utc1 : obspy.UTCDateTime compatible object
        The start time of the entire interval
    utc2 : obspy.UTCDateTime compatible object
        The end time of the entire interval
    duration : float
        The duration of each chunk
    overlap : float
        The overlap each chunk should have (added at end)

    Yields
    ------
    (time1, time2)

    Examples
    --------
    >>> t1 = obspy.UTCDateTime('2016-01-01')
    >>> t2 = t1 + 2 * 3600
    >>> out = list(make_time_chunks(t1, t2, 3600))
    >>> assert out == [(t1, t1 + 3600), (t1 + 3600, t2)]
    """
    utc1 = to_utc(utc1)
    utc2 = to_utc(utc2)
    overlap = overlap or 0.0
    while utc1 < utc2:
        t2 = utc1 + duration + overlap
        if t2 >= utc2 + overlap:
            t2 = utc2 + overlap
        yield utc1, t2
        utc1 += duration  # add duration
