"""
General utilities for obsplus.
"""
import contextlib
import fnmatch
import hashlib
import os
import re
import sys
import textwrap
import warnings
from functools import singledispatch, lru_cache, wraps, partial
from itertools import product
from pathlib import Path
from typing import (
    Union,
    Sequence,
    Optional,
    Any,
    Callable,
    Dict,
    Tuple,
    Generator,
    Set,
    TypeVar,
    Collection,
    Iterable,
)

import numpy as np
import obspy
import obspy.core.event as ev
import pandas as pd
from obspy import UTCDateTime as UTC
from obspy.core.inventory import Station, Channel
from obspy.geodetics.base import gps2dist_azimuth, WGS84_A, WGS84_F
from obspy.io.mseed.core import _read_mseed as mread
from obspy.io.quakeml.core import _read_quakeml
from obspy.taup.taup_geo import calc_dist
from progressbar import ProgressBar

import obsplus
from obsplus.constants import (
    event_time_type,
    NSLC,
    NULL_NSLC_CODES,
    wave_type,
    event_type,
    inventory_type,
    DISTANCE_COLUMNS,
    DISTANCE_DTYPES,
)

BASIC_NON_SEQUENCE_TYPE = (int, float, str, bool, type(None))
# make a dict of functions for reading waveforms
READ_DICT = dict(mseed=mread, quakeml=_read_quakeml)


def deprecated_callable(func=None, replacement_str=None):
    fname = str(getattr(func, "__name__", func))

    if callable(func):

        @wraps(func)
        def _wrap(*args, **kwargs):
            msg = f"{fname} is deprecated and will be removed in a future release."
            if replacement_str:
                msg += f" Please use {replacement_str} instead."
            warnings.warn(msg)
            return func(*args, **kwargs)

        return _wrap
    else:
        return partial(deprecated_callable, replacement_str=replacement_str)


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


def get_instances(*args, **kwargs):
    return [x[0] for x in yield_obj_parent_attr(*args, **kwargs)]


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
    replace
        Input passed to DataFrame.Replace
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

    def wrapper(func):
        dkey = key or func.__name__
        dict[dkey] = func
        return func

    return wrapper


@singledispatch
def get_reference_time(obj: Union[event_time_type, wave_type],) -> obspy.UTCDateTime:
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
        return obspy.UTCDateTime(obj)
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
        assert por.time is not None, f"bad time found on {por}"
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


def to_timestamp(obj: Optional[Union[str, float, obspy.UTCDateTime]], on_none) -> float:
    """
    Convert object to UTC object then get the time stamp.

    If obj is None return on_none value
    """
    if obj is None:
        obj = on_none
    return obspy.UTCDateTime(obj).timestamp


def apply_to_files_or_skip(func: Callable, directory: Union[str, Path]):
    """
    Generator for applying func to all files in directory.

    Skip any files that raise an exception.

    Parameters
    ----------
    func
        Any callable that takes a file path as the only input.

    directory
        A directory that exists.

    Yields
    -------
    outputs of func
    """
    path = Path(directory)
    assert path.is_dir(), f"{directory} is not a directory"
    for fi in path.rglob("*"):
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

    Fails gracefully if bar cannot be displayed (eg if no std out).
    Args and kwargs are passed to ProgressBar constructor.

    Parameters
    ----------
    max_value
        The highest number expected
    min_value
        The minimum number of updates required to show the bar
    """

    def _new_update(bar):
        """ A new update function that swallows attribute and index errors """
        old_update = bar.update

        def update(value=None, force=False, **kwargs):
            try:
                old_update(value=value, force=force, **kwargs)
            except (IndexError, ValueError, AttributeError):
                pass

        return update

    if min_value and max_value < min_value:
        return None  # no progress bar needed, return None
    try:
        bar = ProgressBar(max_value=max_value, *args, **kwargs)
        bar.start()
        bar.update = _new_update(bar)
        bar.update(1)
    except Exception:  # this can happen when stdout is being redirected
        return None  # something went wrong, return None
    return bar


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


def compose_docstring(**kwargs):
    """
    Decorator for composing docstrings.

    Provided values found in curly brackets un wrapped functions docstring
    will be appropriately indented and included in wrapped function's
    __docs__.
    """

    def _wrap(func):

        docstring = func.__doc__
        # iterate each provided value and look for it in the docstring
        for key, value in kwargs.items():
            search_value = "{%s}" % key
            # find all lines that match values
            lines = [x for x in docstring.split("\n") if search_value in x]
            for line in lines:
                # determine number of spaces used before matching character
                spaces = line.split(search_value)[0]
                # ensure only spaces preceed search value
                assert set(spaces) == {" "}
                new = {key: textwrap.indent(textwrap.dedent(value), spaces)}
                docstring = docstring.replace(line, line.format(**new))
        func.__doc__ = docstring
        return func

    return _wrap


def get_seed_id_series(df: pd.DataFrame, null_codes=NULL_NSLC_CODES) -> pd.Series:
    """
    Create a series of seed_ids from a dataframe with required columns.

    The seed id series contains strings of the form:
        network.station.location.channel

    Any "nullish" values (defined by the parameter null_codes) will be
    replaced with an empty string.

    Parameters
    ----------
    df
        Any Dataframe that has columns with str dtype named:
            network, station, location, channel
    null_codes
        Codes which should be replaced with a blank string.

    Returns
    -------
    A series of concatenated seed_ids codes.
    """
    assert set(NSLC).issubset(df.columns), f"dataframe must have columns {NSLC}"
    replace_dict = {x: "" for x in null_codes}
    nslc = df[list(NSLC)].astype(str).replace(replace_dict)
    net, sta, loc, chan = [nslc[x] for x in NSLC]
    return net + "." + sta + "." + loc + "." + chan


@deprecated_callable()
def get_nslc_series(*args, **kwargs) -> pd.Series:
    return get_seed_id_series(*args, **kwargs)


any_type = TypeVar("any_type")


@singledispatch
def replace_null_nlsc_codes(
    obspy_object: any_type, null_codes=NULL_NSLC_CODES, replacement_value=""
) -> any_type:
    """
    Iterate an obspy object and replace nullish nslc codes with some value.

    Operates in place, but also returns the original object.

    Parameters
    ----------
    obspy_object
        An obspy catalog, event, (or any sub element), stream, trace,
        inventory, etc.
    null_codes
        The codes that are considered null values and should be replaced.
    replacement_value
        The value with which to replace the null_codes.
    """
    wid_codes = tuple(x + "_code" for x in NSLC)
    for wid, _, _ in yield_obj_parent_attr(obspy_object, cls=ev.WaveformStreamID):
        for code in wid_codes:
            if getattr(wid, code) in null_codes:
                setattr(wid, code, replacement_value)
    return obspy_object


@replace_null_nlsc_codes.register(obspy.Stream)
def _replace_null_stream(st, null_codes=NULL_NSLC_CODES, replacement_value=""):
    for tr in st:
        _replace_null_trace(tr, null_codes, replacement_value)
    return st


@replace_null_nlsc_codes.register(obspy.Trace)
def _replace_null_trace(tr, null_codes=NULL_NSLC_CODES, replacement_value=""):
    for code in NSLC:
        val = getattr(tr.stats, code)
        if val in null_codes:
            setattr(tr.stats, code, replacement_value)
    return tr


@replace_null_nlsc_codes.register(obspy.Inventory)
@replace_null_nlsc_codes.register(Station)
@replace_null_nlsc_codes.register(Channel)
def _replace_inv_nulls(inv, null_codes=NULL_NSLC_CODES, replacement_value=""):
    for code in ["location_code", "code"]:
        for obj, _, _ in yield_obj_parent_attr(inv, has_attr=code):
            if getattr(obj, code) in null_codes:
                setattr(obj, code, replacement_value)
    return inv


def filter_index(
    index: pd.DataFrame,
    network: Optional = None,
    station: Optional = None,
    location: Optional = None,
    channel: Optional = None,
    starttime: Optional[Union[UTC, float]] = None,
    endtime: Optional[Union[UTC, float]] = None,
    **kwargs,
) -> np.array:
    """
    Filter a waveform index dataframe based on nslc codes and start/end times.

    Parameters
    ----------
    index
        A dataframe to filter which should have the corresponding columns
        to any non-None parameters used in filter.
    network
        A network code as defined by seed standards.
    station
        A station code as defined by seed standards.
    location
        A location code as defined by seed standards.
    channel
        A channel code as defined by seed standards.
    starttime
        The starttime of interest.
    endtime
        The endtime of interest.

    Additional kwargs are used as filters.

    Returns
    -------
    A numpy array of boolean values indicating if each row met the filter
    requirements.
    """
    # handle non-starttime/endtime queries
    query = dict(network=network, station=station, location=location, channel=channel)
    kwargs.update({i: v for i, v in query.items() if v is not None})
    out = filter_df(index, **kwargs)
    # handle starttime/endtime queries if needed
    if starttime is not None or endtime is not None:
        time_out = _filter_starttime_endtime(index, starttime, endtime)
        out = np.logical_and(out, time_out)
    return out


def filter_df(df: pd.DataFrame, **kwargs) -> np.array:
    """
    Determine if each row of the index meets some filter requirements.

    Parameters
    ----------
    df
        The input dataframe.
    kwargs
        Any condition to check against columns of df. Can be a single value
        or a collection of values (to check isin on columns). Str arguments
        can also use unix style matching.

    Returns
    -------
    A boolean array of the same len as df indicating if each row meets the
    requirements.
    """
    # ensure the specified kwarg keys have corresponding columns
    if not set(kwargs).issubset(df.columns):
        msg = f"columns: {set(kwargs) - set(df.columns)} are not found in df"
        raise ValueError(msg)
    # divide queries into flat parameters and collections
    flat_query = {
        k: v
        for k, v in kwargs.items()
        if isinstance(v, str) or not isinstance(v, Collection)
    }
    sequence_query = {
        k: v for k, v in kwargs.items() if k not in flat_query and v is not None
    }
    # get a blank index of True for filters
    bool_index = np.ones(len(df), dtype=bool)
    # filter on non-collection queries
    for key, val in flat_query.items():
        if isinstance(val, str):
            regex = get_regex(val)
            new = df[key].str.match(regex).values
            bool_index = np.logical_and(bool_index, new)
        else:
            new = (df[key] == val).values
            bool_index = np.logical_and(bool_index, new)
    # filter on collection queries using isin
    for key, val in sequence_query.items():
        bool_index = np.logical_and(bool_index, df[key].isin(val))

    return bool_index


def _filter_starttime_endtime(df, starttime=None, endtime=None):
    """ Filter dataframe on starttime and endtime. """
    bool_index = np.ones(len(df), dtype=bool)
    t1 = UTC(starttime).timestamp if starttime is not None else -1 * np.inf
    t2 = UTC(endtime).timestamp if endtime is not None else np.inf
    # get time columns
    start_col = getattr(df, "starttime", getattr(df, "start_date", None))
    end_col = getattr(df, "endtime", getattr(df, "end_date", None))
    in_time = ~((end_col < t1) | (start_col > t2))
    return np.logical_and(bool_index, in_time.values)


def iter_files(path, ext=None, mtime=None, skip_hidden=True):
    """
    use os.scan dir to iter files, optionally only for those with given
    extension (ext) or modified times after mtime

    Parameters
    ----------
    path : str
        The path to the base directory to traverse.
    ext : str or None
        The extensions to map.
    mtime : int or float
        Time stamp indicating the minimum mtime.
    skip_hidden : bool
        If True skip files that begin with a '.'

    Returns
    -------

    """
    for entry in os.scandir(path):
        if entry.is_file() and (ext is None or entry.name.endswith(ext)):
            if mtime is None or entry.stat().st_mtime >= mtime:
                if entry.name[0] != "." or not skip_hidden:
                    yield entry.path
        elif entry.is_dir():
            yield from iter_files(
                entry.path, ext=ext, mtime=mtime, skip_hidden=skip_hidden
            )


def get_distance_df(
    entity_1: Union[event_type, inventory_type],
    entity_2: Union[event_type, inventory_type],
    a: float = WGS84_A,
    f: float = WGS84_F,
) -> pd.DataFrame:
    """
    Create a dataframe of distances and azimuths from events to stations.

    Parameters
    ----------
    entity_1
        An object from which latitude, longitude and depth are
        extractable.
    entity_2
        An object from which latitude, longitude and depth are
        extractable.
    a
        Radius of planetary body (usually Earth) in m. Defaults to WGS84.
    f
        Flattening of planetary body (usually Earth). Defaults to WGS84.

    Notes
    -----
    Simply uses obspy.geodetics.gps2dist_azimuth under the hood. The index
    of the dataframe is a multi-index where the first level corresponds to ids
    from entity_1 (either an event_id or seed_id str) and the second level
    corresponds to ids from entity_2.

    Returns
    -------
    A dataframe with distance, horizontal_distance, and depth distances,
    as well as azimuth, for each entity pair.
    """

    def _dist_func(tup1, tup2):
        """
        Function to get distances and azimuths from pairs of coordinates
        """
        gs_dist, azimuth = gps2dist_azimuth(*tup2[:2], *tup1[:2], a=a, f=f)[:2]
        z_diff = tup2[2] - tup1[2]
        dist = np.sqrt(gs_dist ** 2 + z_diff ** 2)
        out = dict(
            horizontal_distance=gs_dist,
            distance=dist,
            depth_distance=z_diff,
            azimuth=azimuth,
        )
        return out

    def _get_distance_tuple(obj):
        """ return a list of tuples for entities """
        cols = ["latitude", "longitude", "elevation", "id"]
        try:
            df = obsplus.events_to_df(obj)
            df["elevation"] = -df["depth"]
            df["id"] = df["event_id"]
        except (TypeError, ValueError, AttributeError):
            df = obsplus.stations_to_df(obj)
            df["id"] = df["seed_id"]
        return set(df[cols].itertuples(index=False, name=None))

    # get a tuple of
    coord1 = _get_distance_tuple(entity_1)
    coord2 = _get_distance_tuple(entity_2)
    # make a dict of {(event_id, seed_id): (dist, azimuth)}
    dist_dicts = {
        (ell[-1], ill[-1]): _dist_func(ell, ill)
        for ell, ill in product(coord1, coord2)
        if ell[-1] != ill[-1]  # skip if same entity
    }
    df = pd.DataFrame(dist_dicts).T.astype(DISTANCE_DTYPES)
    # make sure index is named
    df.index.names = ("id1", "id2")
    return df[list(DISTANCE_COLUMNS)]


def calculate_distance(
    latitude: float,
    longitude: float,
    df,
    degrees: bool = True,
    a: float = WGS84_A,
    f: float = WGS84_F,
) -> pd.Series:
    """
    Calculate the distance from all events in the dataframe to a set point.

    Parameters
    ----------
    latitude
        Latitude in degrees for point to calculate distance from
    longitude
        Longitude in degrees for point to calculate distance from
    df
        DataFrame to compute distances for. Must have columns titles
        "latitude" and "longitude"
    degrees
        Whether to return distance in degrees (default) or in meters.
    a
        Radius of planetary body (usually Earth) in m. Defaults to WGS84.
    f
        Flattening of planetary body (usually Earth). Defaults to WGS84.

    Returns
    -------
    A series of distances (in degrees if `degrees=True` or meters) indexed in
    the same way as the input dataframe.
    """
    if latitude > 90 or latitude < -90:
        raise ValueError("Latitude of Point 1 out of bounds! (-90 <= lat1 <=90)")
    _a = a / 1000  # calc_dist needs this in km, but other things use m - convert once.

    def _degrees_dist_func(_df):
        if _df["latitude"] > 90 or _df["latitude"] < -90:
            raise ValueError(
                "Latitude in dataframe out of bounds! "
                "(-90 <= {0} <=90)".format(_df["latitude"])
            )
        return calc_dist(
            source_latitude_in_deg=latitude,
            source_longitude_in_deg=longitude,
            receiver_latitude_in_deg=_df["latitude"],
            receiver_longitude_in_deg=_df["longitude"],
            radius_of_planet_in_km=_a,
            flattening_of_planet=f,
        )

    def _km_dist_func(_df):
        dist, _, _ = gps2dist_azimuth(
            lat1=latitude, lon1=longitude, lat2=_df["latitude"], lon2=_df["longitude"]
        )
        return dist

    if degrees:
        _dist_func = _degrees_dist_func
    else:
        _dist_func = _km_dist_func

    return df.apply(_dist_func, axis=1, result_type="reduce")


@lru_cache(maxsize=2500)
def get_regex(nslc_str):
    """ Compile, and cache regex for str queries. """
    return fnmatch.translate(nslc_str)  # translate to re


def md5(path: Union[str, Path]):
    """
    Calculate the md5 hash of a file.

    Reads the file in chunks to allow using large files. Taken from this stack
    overflow answer: http://bit.ly/2Jqb1Jr

    Parameters
    ----------
    path
        The path to the file to read.

    Returns
    -------
    A str of hex for file hash

    """
    path = Path(path)
    hash_md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def md5_directory(
    path: Union[Path, str],
    match: str = "*",
    exclude: Optional[Union[str, Collection[str]]] = None,
    hidden=False,
) -> Dict[str, str]:
    """
    Calculate the md5 hash of all files in a directory.

    Parameters
    ----------
    path
        The path to the directory
    match
        A unix-style matching string
    exclude
        A list of unix style strings to exclude
    hidden
        If True skip all files starting with a .

    Returns
    -------
    A dict containing paths and md5 hashes.
    """
    path = Path(path)
    out = {}
    excludes = iterate(exclude)
    for sub_path in path.rglob(match):
        keep = True
        # skip directories
        if sub_path.is_dir():
            continue
        # skip if matches on exclusion
        for exc in excludes:
            if fnmatch.fnmatch(sub_path.name, exc):
                keep = False
                break
        if not hidden and sub_path.name.startswith("."):
            keep = False
        if keep:
            out[str(sub_path.relative_to(path))] = md5(sub_path)
    return out


def _column_contains(ser: pd.Series, str_sequence: Iterable[str]) -> pd.Series:
    """ Test if a str series contains any values in a sequence """
    safe_matches = {re.escape(x) for x in str_sequence}
    return ser.str.contains("|".join(safe_matches)).values
