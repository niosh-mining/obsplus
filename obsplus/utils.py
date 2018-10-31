# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 06:29:04 2016

@author: amy
"""
import contextlib
import glob
import os
import sys
import threading
import warnings
from functools import singledispatch, wraps
from typing import Union, Sequence, Optional, Any, Callable, Dict, Tuple, Generator, Set

import numpy as np
import obspy
import obspy.core.event as ev
import pandas as pd
from obspy.core.event import Event
from obspy.core.inventory import Inventory
from obspy.io.mseed.core import _read_mseed as mread
from obspy.io.quakeml.core import _read_quakeml
from progressbar import ProgressBar

from obsplus.constants import event_time_type

BASIC_NON_SEQUENCE_TYPE = (int, float, str, bool, type(None))
# make a dict of functions for reading waveforms
READ_DICT = dict(mseed=mread, quakeml=_read_quakeml)


# ------------------------------ wavebank stuff


@singledispatch
def get_inventory(inventory: Union[str, Inventory]):
    """
    Get an stations from stations parameter if path or stations else
    return None

    Parameters
    ----------
    inventory : str, obspy.Inventory, or None

    Returns
    -------
    obspy.Inventory or None
    """
    assert isinstance(inventory, Inventory) or inventory is None
    return inventory


@get_inventory.register(str)
def _get_inv_str(inventory):
    """ if str is provided """
    return obspy.read_inventory(inventory)


# ---------------------------------- Misc functions


def get_instances(
    obj: object,
    cls: type = None,
    is_attr: Optional[str] = None,
    has_attr: Optional[str] = None,
) -> list:
    """
    Recurse object, return a list of instances of meeting search criteria.

    Parameters
    ----------
    obj
        The object to recurse through attributes of lists, tuples, and other
        instances.
    cls
        Only return instances of cls if not None, else return all instances.
    is_attr
        Only return objects stored as attr_name, if None return all.
    has_attr
        Only return objects that have attribute has_attr, if None return all.
    """
    instance_cache: dict = {}

    def _get_instances(obj, cls, ids=None, attr=None):
        out = []
        ids = ids or set()  # id cache to avoid circular references
        if id(obj) in ids:
            return []
        if (id(obj), cls) in instance_cache:
            return instance_cache[(id(obj), cls)]
        ids.add(id(obj))
        if cls is None or isinstance(obj, cls):
            # filter out built-ins by looking for __dict__ or __slots__
            not_bultin = hasattr(obj, "__dict__") or hasattr(obj, "__slots__")
            # check if this object is stored as the desired attribute
            is_attribute = is_attr is None or attr == is_attr
            # check if object has desired attribute
            has_attribute = has_attr is None or hasattr(obj, has_attr)
            if not_bultin and is_attribute and has_attribute:
                out.append(obj)
        if hasattr(obj, "__dict__"):
            for item, val in obj.__dict__.items():
                out += _get_instances(val, cls, ids, attr=item)
        if isinstance(obj, (list, tuple)):
            for val in obj:
                out += _get_instances(val, cls, ids, attr=attr)
        instance_cache[(id(obj), cls)] = out
        return out

    return _get_instances(obj, cls)


def yield_obj_parent_attr(
    obj, cls=None, is_attr=None, has_attr=None, basic_types=False
) -> Generator[Tuple[Any, Any, str], None, None]:
    """
    Recurse an object, yield a tuple of object, parent, attr.

    Can be used, for example, to yield all ResourceIdentifier instances
    contained in any obspy.core.event class instances and attached instances,
    as well as the objects they are attached to (parents) and the attribute
    name in which they are stored (attr).

    Parameters
    -----------
    obj
        The object to recurse through attributes of lists, tuples, and other
        instances.
    cls
        Only return instances of cls if not None, else return all instances.
    is_attr
        Only return objects stored as attr_name, if None return all.
    has_attr
        Only return objects that have attribute has_attr, if None return all.
    basic_types
        If True, yield non-sequence basic types (int, float, str, bool).
    """
    ids: Set[int] = set()  # id cache to avoid circular references

    def func(obj, attr=None, parent=None):
        id_tuple = (id(obj), id(parent))

        # If object/parent combo have not been yielded continue.
        if id_tuple in ids:
            return
        ids.add(id_tuple)
        # Check if this object is stored as the desired attribute.
        is_attribute = is_attr is None or attr == is_attr
        # Check if the object has the desired attribute.
        has_attribute = has_attr is None or hasattr(obj, has_attr)
        # Check if isinstance of desired class.
        is_instance = cls is None or isinstance(obj, cls)
        # Check if basic type (dont
        is_basic = basic_types or not isinstance(obj, BASIC_NON_SEQUENCE_TYPE)
        # Iterate through basic built-in types.
        if isinstance(obj, (list, tuple)):
            for val in obj:
                yield from func(val, attr=attr, parent=parent)
        elif isinstance(obj, dict):
            for item, val in obj.items():
                yield from func(val, attr=item, parent=obj)
        # Yield object, parent, and attr if desired conditions are met.
        elif is_attribute and has_attribute and is_instance and is_basic:
            yield (obj, parent, attr)
        # Iterate through non built-in object attributes.
        if hasattr(obj, "__slots__"):
            for attr in obj.__slots__:
                val = getattr(obj, attr)
                yield from func(val, attr=attr, parent=obj)
        if hasattr(obj, "__dict__"):
            for item, val in obj.__dict__.items():
                yield from func(val, attr=item, parent=obj)

    return func(obj)


def make_time_chunks(
    utc1, utc2, duration, overlap=0.0
) -> Generator[Tuple[obspy.UTCDateTime, obspy.UTCDateTime], None, None]:
    """
    Yield time intervals fitting in given datetime range.

    Function to take two utc date time objects and create a generator to
    yield all time in between by intervals of duration. Overlap is number
    of seconds at end of file
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
    -------
    (time1, time2)

    Examples
    ----------
    >>> t1 = obspy.UTCDateTime('2016-01-01')
    >>> t2 = t1 + 3 * 3600
    >>> out = make_time_chunks(t1, t2, 3600)
    >>> assert out == [t1, t1 + 3600, t2]
    """
    utc1 = obspy.UTCDateTime(utc1)
    utc2 = obspy.UTCDateTime(utc2)
    overlap = overlap or 0.0
    while utc1 < utc2:
        t2 = utc1 + duration + overlap
        if t2 >= utc2 + overlap:
            t2 = utc2 + overlap
        yield (utc1, t2)
        utc1 += duration  # add duration


def try_read_catalog(catalog_path, **kwargs):
    """ Try to read a events from file, if it raises return None """
    read = READ_DICT.get(kwargs.pop("format", None), obspy.read_events)
    try:
        cat = read(catalog_path, **kwargs)
    except Exception:
        warnings.warn(f"obspy failed to read {catalog_path}")
    else:
        if cat is not None and len(cat):
            return cat
    return None


def order_columns(
    df: pd.DataFrame,
    required_columns: Sequence,
    dtype: Optional[Dict[str, type]] = None,
    replace: Optional[dict] = None,
):
    """
    Given a dataframe, assert that required columns are in the df, then
    order the columns of df the same as required columns with extra columns
    attached at the end.

    Parameters
    ----------
    df
        A dataframe
    required_columns
        A sequence that contains the column names
    dtype
        A dictionary of dtypes
    Returns
    -------
    pd.DataFrame
    """
    if df.empty:
        return pd.DataFrame(columns=required_columns)
    # add any extra columns if needed
    if not set(df.columns).issuperset(required_columns):
        df = df.reindex(columns=required_columns)
    # make sure required columns are there
    column_set = set(df.columns)
    extra_cols = sorted(list(column_set - set(required_columns)))
    new_cols = list(required_columns) + extra_cols
    # cast network, station, location, channel, to str
    if dtype:
        used = set(dtype) & set(df.columns)
        df = df.astype({i: dtype[i] for i in used})
    if replace:
        try:
            df = df.replace(replace)
        except Exception:
            pass
    assert column_set == set(new_cols)
    return df[new_cols]


def read_file(file_path, funcs=(pd.read_csv,)) -> Optional[Any]:
    """
    For a given file_path, try reading it with each function in funcs.

    Parameters
    ----------
    file_path
        The path to the file to read
    funcs
        A tuple of functions to try to read the file (starting with firsts)

    """
    for func in funcs:
        assert callable(func)
        try:
            return func(file_path)
        except Exception:
            pass
    raise IOError(f"failed to read {file_path}")


def register_func(dict, key=None):
    """ decorator to register a function in a list or dict """

    def wraper(func):
        dkey = key or func.__name__
        dict[dkey] = func
        return func

    return wraper


@singledispatch
def get_reference_time(obj: event_time_type) -> obspy.UTCDateTime:
    """
    Get a refernce time inferred from an object.

    Parameters
    ----------
    obj
        The argument that will indicate a start time. Can be a one length
        events, and event, a float, or a UTCDatetime object

    Returns
    -------
    obspy.UTCDateTime
    """
    if obj is None:
        return None
    return obspy.UTCDateTime(obj)


@get_reference_time.register(obspy.core.event.Event)
def _get_event_origin_time(event):
    """ get the time from preferred origin from the event """
    # try to get origin
    try:
        por = event.preferred_origin() or event.origins[-1]
    except IndexError:
        por = None
    if por is not None:
        assert por.time is not None, f"bad time found on {por}"
        return get_reference_time(por.time)
    # else try using picks
    elif event.picks:
        return get_reference_time(event.picks)
    else:
        msg = "could not get reference time for {event}"
        raise ValueError(msg)


@get_reference_time.register(ev.Pick)
def _get_first_pick(pick):
    """ ensure the events is length one, return event """
    return get_reference_time(pick.time)


@get_reference_time.register(list)
def _from_list(input_list):
    """ ensure the events is length one, return event """
    outs = [get_reference_time(x) for x in input_list]
    return min([x for x in outs if x is not None])


@get_reference_time.register(obspy.Catalog)
def _get_first_event(catalog):
    """ ensure the events is length one, return event """
    assert len(catalog) == 1, f"{events} has more than one event"
    return _get_event_origin_time(catalog[0])


def get_preferred(event: Event, what: str):
    """
    get the preferred what (eg origin, magnitude) from the event.
    If not defined use the last in the list.
    If list is empty init empty object.
    Parameters
    -----------
    event: obspy.core.event.Event
        The instance for which the preferred should be sought
    what: the prefered item to get
        eg, magnitude, origin, focal_mechanism
    """
    prefname = "preferred_" + what
    whats = what + "s"
    obj = getattr(event, prefname)()
    if obj is None:  # get end of list
        pid = getattr(event, prefname + "_id")
        if pid is None:  # no preferred id set, return last in list
            try:  # if there is None return an empty one
                obj = getattr(event, whats)[-1]
            except IndexError:  # object has no whats (eg magnitude)
                return getattr(obspy.core.event, what.capitalize())()
        else:  # there is an id, it has just come detached, try to find it
            potentials = {x.resource_id.id: x for x in getattr(event, whats)}
            if pid.id in potentials:
                obj = potentials[pid.id]
            else:
                var = (pid.id, whats, str(event))
                warnings.warn("cannot find %s in %s for event %s" % var)
                obj = getattr(event, whats)[-1]
    return obj


def to_timestamp(obj: Optional[Union[str, float, obspy.UTCDateTime]], on_none) -> float:
    """
    Convert object to UTC object then get the time stamp.

    If obj is None return on_none value
    """
    if obj is None:
        obj = on_none
    return obspy.UTCDateTime(obj).timestamp


def apply_or_skip(func: Callable, directory: str):
    """
    Generator for applying func to all files in directory.

    Skip any files that raise an exception.

    Parameters
    ----------
    func
        Any callable that takes a file path as the only input

    directory
        A directory that exists

    Yields
    -------
    outputs of func
    """
    assert os.path.isdir(directory), f"{directory} is not a directory"
    for fi in glob.iglob(os.path.join(directory, "**", "*"), recursive=True):
        if os.path.isfile(fi):
            try:
                yield func(fi)
            except Exception:
                pass


def get_progressbar(
    max_value, min_value=None, *args, **kwargs
) -> Optional[ProgressBar]:
    """
    Get a progress bar object using the ProgressBar2 library.

    Fails gracefully if bar cannot be displayed (eg if not std out).
    Args and kwargs are passed to ProgressBar constructor.

    Parameters
    ----------
    max_value
        The highest number expected
    min_value
        The minimum number of updates required to show the bar
    """

    if min_value and max_value < min_value:
        return None  # no progress bar needed, return None
    try:
        bar = ProgressBar(max_value=max_value, *args, **kwargs)
        bar.start()
        bar.update(1)
    except Exception:
        return None  # something went wrong, return None
    return bar


def thread_lock_function(lock=None):
    """
    A decorator for locking a function that should never be run by more than
    one thread.

    Parameters
    ----------
    lock
        An RLock or Lock object from the threading module, or an object that
        has the same interface.
    """
    lock = lock or threading.RLock()

    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)

        return _wrapper

    return _decorator


def iterate(obj):
    """
    Return an iterable from any object.

    If string, do not iterate characters, return str in tuple .
    """
    if obj is None:
        return ()
    if isinstance(obj, str):
        return (obj,)
    return obj if isinstance(obj, Sequence) else (obj,)


class DummyFile(object):
    """ Dummy class to mock std out interface but go nowhere. """

    def write(self, x):
        """ do nothing """

    def flush(self):
        """ do nothing """


@contextlib.contextmanager
def no_std_out():
    """
    Silence std out.
    Taken from here: goo.gl/eVx6oj
    """
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def getattrs(obsject, col_set, default_value=np.nan):
    """
    Parse an object for a list of attrs, return a dict of values or None
    """
    out = {}
    if obsject is None:  # return empty if None
        return out
    for item in col_set:
        try:
            val = getattr(obsject, item)
        except (ValueError, AttributeError):
            val = default_value
        if val is None:
            val = default_value
        out[item] = val
    return out
