"""
Functions that were too small to put into their own module
"""

import copy
import logging
import os
import re
import warnings
from functools import lru_cache, singledispatch
from pathlib import Path
from typing import Union, Optional, Callable, Iterable

import obspy
import obspy.core.event as ev
import pandas as pd
from obspy.core.event import Catalog, Event, ResourceIdentifier, WaveformStreamID
from obspy.core.event.base import QuantityError
from obspy.core.util.obspy_types import Enum

import obsplus
from obsplus.constants import (
    UTC_FORMATS,
    catalog_or_event,
    catalog_component,
    EVENT_ATTRS,
    UTC_KEYS,
    event_clientable_type,
)
from obsplus.interfaces import EventClient
from obsplus.utils import yield_obj_parent_attr, get_reference_time, get_nslc_series


def duplicate_events(
    event: catalog_or_event, resource_generator: Optional[Callable] = None
) -> catalog_or_event:
    """
    Duplicate an event.

    Creates a copy of the event, and assigns new resource IDs to all
    internal objects (ie picks, station_magnitudes, etc.). Will not
    assign new resource_id to those representing external resources
    (eg analyst, velocity models, etc.).

    Parameters
    ----------
    event
        The event to duplicate.
    resource_generator:
        A callable that can be used to generate resource IDs.
    """
    new = copy.deepcopy(event)
    gen_func = resource_generator or ResourceIdentifier
    id_map = {}  # map old id to new
    # first pass through bind resource id to parent if attr is resource_id
    for rid, parent, attr in yield_obj_parent_attr(new, ResourceIdentifier):
        if attr == "resource_id":
            try:
                prefix = rid.prefix
            except AttributeError:
                prefix = "smi:local"
            new_rid = gen_func(referred_object=parent, prefix=prefix)
            id_map[rid.id] = new_rid
            setattr(parent, attr, new_rid)
    # second pass, swap out resource ids with the same id as those in id_map
    for rid, parent, attr in yield_obj_parent_attr(new, ResourceIdentifier):
        if rid.id in id_map and attr != "resource_id":
            setattr(parent, attr, id_map[rid.id])
    return new


def catalog_to_directory(
    cat: Union[str, Catalog, ev.Event, Path],
    path: Union[str, Path],
    file_format: str = "quakeml",
    ext="xml",
    event_bank_index=True,
) -> None:
    """
    Parse a catalog and save each event to a time-based directory structure.

    Saves each event based on preferred origin time. The format of the saved
    file and directory is:
        YYYY/MM/DD/YYYY-MM-DDThh-mm-ss-RID
    where RID is the last 5 digits of the event id. If another event is found
    with the same last 5 digits of the resource_id the event will be read into
    memory. If the complete resource IDs are the same the old path will be
    used. This helps avoid changing the path of the event when origin times
    change slightly.

    Parameters
    ----------
    cat
        The obspy events, event or path to such.
    path
        A path to the directory. If one does not exist it will be created.
    file_format.
        Any obspy event format that can be written.
    ext
        The extention to add to each file.
    event_bank_index
        If True, create an event bank index on the newly created directory.
    """
    if isinstance(cat, (str, Path)):
        cat = obspy.read_events(str(cat))
    # ensure events is iterable and ext has a dot before it
    cat = [cat] if not isinstance(cat, obspy.Catalog) else cat
    ext = "." + ext if not ext.startswith(".") else ext
    # make sure directory exists
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # iterate each event, get a time and resource id and save to disk
    for event in cat:
        event_path = Path(get_event_path(event, str(path), ext=ext))
        path.parent.mkdir(parents=True, exist_ok=True)
        # determine if another event exists with same id, if so use its path
        rid = str(event.resource_id)[-5:]
        possible_duplicate = list(event_path.parent.glob(f"*{rid}{ext}")) or []
        for duplicate_path in possible_duplicate:
            new_event = obspy.read_events(str(duplicate_path))[0]
            if new_event.resource_id == event.resource_id:
                event_path = duplicate_path
                break
        event.write(str(event_path), file_format)
    if event_bank_index:
        obsplus.EventBank(path).update_index()


def prune_events(events: catalog_or_event) -> Catalog:
    """
    Remove all the unreferenced rejected objects from an event or catalog.

    This function first creates a copy of the event/catalog. Then it looks
    all objects with rejected evaluation status', which are not referred to
    by any unrejected object, and removes them.

    Parameters
    ----------
    events
        The events to iterate and modify

    Returns
    -------
    A Catalog with the pruned events.

    """
    from obsplus.events.validate import validate_catalog

    # ensure we have something iterable
    events = [events] if isinstance(events, ev.Event) else events
    out = []

    def _get_edges_rids_opa(event):
        """
        Return a list of edges (resource_id_parent, resource_id_child),
        a set of rejected resource_ids and a dict of
        {resource_id: (obj, parent, attr)}.
        """

        # {status: [obj, ]}
        edges = []  # list of tuples (parent, child)
        # {resource_id, }
        rejected_rid = set()
        # {resource_id: [obj, parent, attr]}
        rejected_opa = {}
        # first make a list objects with eval status as well as all other
        # objects they refer to, using resource_ids
        opa_iter = yield_obj_parent_attr(event, has_attr="evaluation_status")
        for obj, parent, attr in opa_iter:
            rid = str(obj.resource_id)
            is_rejected = obj.evaluation_status == "rejected"
            if is_rejected:
                rejected_rid.add(rid)
                rejected_opa[rid] = (obj, parent, attr)
            # now recurse object and find all objects this one refers to
            opa_iter_2 = yield_obj_parent_attr(obj, cls=ev.ResourceIdentifier)
            for obj2, _, _ in opa_iter_2:
                edges.append((rid, str(obj2)))
        return edges, rejected_rid, rejected_opa

    def _remove_object(obj, parent, attr):
        """
        Remove the object.
        """
        maybe_obj = getattr(parent, attr)
        if maybe_obj is obj:
            setattr(parent, attr, None)
        else:
            assert isinstance(maybe_obj, list)
            maybe_obj.remove(obj)

    # iterate events, find and destroy rejected orphans (that sounds bad...)
    for event in events:
        event = event.copy()
        validate_catalog(event)
        import obsplus

        obsplus.debug = True
        edges, rejected_rid, rejected_opa = _get_edges_rids_opa(event)
        df = pd.DataFrame(edges, columns=["parent", "child"])
        # filter out non rejected children
        df = df[df.child.isin(rejected_opa)]
        # iterate rejected children IDs and search for any non-rejected parent
        for rid, dff in df.groupby("child"):
            if dff.parent.isin(rejected_rid).all():
                _remove_object(*rejected_opa[rid])
        out.append(event)

    return Catalog(out)


def bump_creation_version(obj):
    """
    Bump the version in an object's CreationInfo and add creation time.

    Take an obspy object and try to bump creation info. This is done
    by first creating info if obj.creation_info is None, then addition
    creation_time and bumping the version by one
    """
    if not hasattr(obj, "creation_info"):  # nothing to do if no CI
        msg = "%s has no creation_info attribute" % obj
        logging.warning(msg)
        return
    # get creation info
    now = obspy.UTCDateTime.now()
    ci = obj.creation_info
    if ci is None:
        ci = obspy.core.event.CreationInfo()
    ci.creation_time = now
    ci.version = _bump_version(ci.version)
    obj.creation_info = ci


def _bump_version(version):
    """ bump the version in the creation info"""
    if isinstance(version, str):
        split = [int(x) for x in version.split(".")]
        split[-1] += 1
        version = ".".join([str(x) for x in split])
    elif version is None:
        version = "0.0.0"
    return version


def make_origins(
    events: catalog_or_event,
    inventory: obspy.Inventory,
    depth: float = 1.0,
    phase_hints: Optional[Iterable] = ("P", "p"),
) -> catalog_or_event:
    """
    Iterate a catalog or single events and ensure each has an origin.

    If no origins are found for an event, create one with the time set to
    the earliest pick and the location set to the location of the first hit
     station. Events are modified in place.

    This may be useful for location codes that need a starting location.

    Parameters
    ----------
    events
        The events to scan and add origins were necessary.
    inventory
        An inventory object which contains all the stations referenced in
        quakeml elements of events.
    depth
        The default depth for created origins. Should be in meters. See the
        obspy docs for Origin or the quakeml standard for more details.
    phase_hints
        List of acceptable phase hints to use for identifying the earliest
        pick. By default will only search for "P" or "p" phase hints.

    Returns
    -------
    Either a Catalog or Event object (same as input).
    """
    # ensure input is an iterable of events
    cat = [events] if isinstance(events, Event) else events
    # load inv dataframe and make sure it has a seed_id column
    df = obsplus.stations_to_df(inventory)
    nslc_series = get_nslc_series(df)
    for event in cat:
        if not event.origins:  # make new origin
            picks = event.picks_to_df()
            picks = picks.loc[
                (~(picks.evaluation_status == "rejected"))
                & (picks.phase_hint.isin(phase_hints))
            ]
            if not len(picks):
                raise ValueError(f"{event} has no acceptable picks to create origin")
            # get first pick, determine time/station used
            first_pick = picks.loc[picks.time == picks.time.min()].iloc[0]
            seed_id = first_pick.seed_id
            # find channel corresponding to pick
            df_chan = df[nslc_series == seed_id]
            if not len(df_chan):
                raise KeyError(f"{seed_id} not found in inventory")
            ser = df_chan.iloc[0]
            # create origin
            ori = _create_first_pick_origin(first_pick, ser, depth=depth)
            event.origins.append(ori)
    return events


def _create_first_pick_origin(first_pick, channel_ser, depth):
    """ Create an origin based on first pick and a channel series. """
    msg = (
        "origin fixed to location and time of earliest pick by "
        f"obsplus version {obsplus.__version__}"
    )
    comment = ev.Comment(text=msg)
    odict = dict(
        time=obspy.UTCDateTime(first_pick.time),
        latitude=channel_ser.latitude,
        longitude=channel_ser.longitude,
        depth=depth,
        time_fixed=True,
        comments=[comment],
    )
    return ev.Origin(**odict)


def get_event_path(
    eve: Union[Catalog, Event],
    base_directory: str = ".",
    create_directories: bool = True,
    ext=".xml",
) -> str:
    """
    Get a path for the event (or len(1) catalog) to save to disk.

    This function is used in internal NIOSH systems and as such is very
    opinionated about how events should be named. It does the following:
    The comments will first be scanned for seisan comments, identified
    with the "ID:" tag. If one is found the date string used in the seisan
    comment will be used for the file name. If a comment with the substring
    "ID:" is not found the name will be generated based on the preferred
    origin time.

    The last 5 characters of the event ID will be included in the file name.

    Parameters
    ----------
    eve : obspy.Event
        The event to get a path for
    base_directory : str
        The path to the base directory
    create_directories: bool
        If True create any directories needed for path that do not exist
    ext : str
        The extension for the file to save

    Returns
    -------
    str :
        The path

    """
    if isinstance(eve, Catalog):
        assert len(eve) == 1, "events must have only one event"
        eve = eve[0]
    utc = get_reference_time(eve)
    path = get_utc_path(utc, base_dir=base_directory, create=create_directories)
    fname = _get_comment_path(eve, ext) or _get_file_name_from_event(eve, ext)
    return os.path.join(path, fname)


def get_utc_path(
    utc: obspy.UTCDateTime,
    base_dir: str,
    format: str = "year/month/day",
    create: bool = True,
):
    """
    Get a subdirectory structure from a UTCDateTime object according to format.

    Parameters
    ----------
    utc : obspy.UTCDateTime
        The time to use in the path creation
    base_dir : str
        The base directory
    format : str
        The format to return
    create : bool
        If True create the directory structure

    Returns
    -------

    """
    selected_fmts = format.split("/")
    assert set(selected_fmts).issubset(UTC_FORMATS)
    utc_list = [UTC_FORMATS[x] % getattr(utc, x) for x in selected_fmts]
    out = os.path.join(*utc_list)
    if base_dir:  # add base dir to start
        out = os.path.join(base_dir, out)
    if create and not os.path.exists(out):  # create dir/subdirs
        os.makedirs(out)
    return out


def get_seed_id(obj: catalog_component) -> str:
    """
    Get the SCNL associated with a station-specific object

    Parameters
    ----------
    obj
        The object for which to retrieve the SCNL. Can be anything that
        has a waveform_id attribute or refers to an object with a
        waveform_id attribute.

    Returns
    -------
    str :
        The SCNL, in the form of a seed string
    """
    if isinstance(obj, WaveformStreamID):
        # Get the seed id
        return obj.get_seed_string()
    if isinstance(obj, ResourceIdentifier):
        # Get the next nested object
        return get_seed_id(obj.get_referred_object())
    # The order of this list matters... it should go from the deepest nested
    # attribute to the shallowest
    attrs = ["waveform_id", "pick_id", "amplitude_id", "station_magnitude_id"]
    # Make sure the object has at least one of the required attributes
    if not len(set(attrs).intersection(set(vars(obj)))):
        raise TypeError(f"cannot retrieve seed id for objects of type {type(obj)}")
    # Loop over each of the attributes, if it exists and is not None,
    # go down a level until it finds a seed id
    for att in attrs:
        if hasattr(obj, att):
            val = getattr(obj, att)
            if val:
                try:
                    return get_seed_id(val)
                except (TypeError, AttributeError):
                    raise AttributeError(
                        "Unable to fetch a seed id for {obj.resource_id}"
                    )
    # If it makes it this far, it could not find a non-None attribute
    raise AttributeError("Unable to fetch a seed id for {obj.resource_id}")


def _get_file_name_from_event(eve: Event, ext: str = ".xml") -> str:
    """
    Generate a file name for the given event.

    Parameters
    ----------
    eve
    ext

    Returns
    -------
    """
    # get utc and formated list from event
    utc = get_reference_time(eve)
    fmt = "year month day hour minute second".split()
    utc_list = [UTC_FORMATS[x] % getattr(utc, x) for x in fmt]
    # append last 5 of resource_id
    rid_str = str(eve.resource_id)[-5:]
    utc_list.append(rid_str)
    fn = "%s-%s-%sT%s-%s-%s-%s" % tuple(utc_list)
    if ext:
        fn = fn + ext
    return fn


def _get_comment_path(eve, ext=".xml"):
    """ scan comments, if any have the ID: tag use the following lines
    to name the file (ID is added by seisan)"""
    for comment in eve.comments:
        txt = comment.text
        if " ID:" in txt:
            ymdhms = txt.split("ID:")[-1][:14]
            year = ymdhms[:4]
            month = ymdhms[4:6]
            dayhour = ymdhms[6:8] + "T" + ymdhms[8:10]
            minute = ymdhms[10:12]
            second = ymdhms[12:14]
            rid = eve.resource_id.id[-5:] + ext
            return "-".join([year, month, dayhour, minute, second, rid])


def _get_params_from_docs(obj):
    """ Attempt to figure out params for obj from the doc strings """
    doc_list = obj.__doc__.splitlines(keepends=False)
    params_lines = [x for x in doc_list if ":param" in x]
    params = [x.split(":")[1].replace("param ", "") for x in params_lines]
    return params


def _getattr_factory(attrs_to_get):
    """ return a function that tries to get attrs into dict """

    def func(obj):
        out = {x: getattr(obj, x) for x in attrs_to_get if hasattr(obj, x)}
        return out or None  # return None rather than empty dict

    return func


def _get_str(obj):
    """ return str of obj """
    return str(obj)


_TO_DICT_FUNCS = {obspy.UTCDateTime: _get_str, Event: _getattr_factory(EVENT_ATTRS)}


@singledispatch
def get_event_client(events: event_clientable_type) -> EventClient:
    """
    Extract an event client from various inputs.

    Parameters
    ----------
    events
        Any of the following:
            * A path to an obspy-readable event file
            * A path to a directory of obspy-readable event files
            * An `obspy.Catalog` instance
            * An instance of :class:`~obsplus.EventBank`
            * Any other object that has a `get_events` method

    Raises
    ------
    TypeError
        If an event client cannot be determined from the input.
    """
    if not isinstance(events, EventClient):
        msg = f"an event client could not be extracted from {events}"
        raise TypeError(msg)
    return events


@get_event_client.register(str)
@get_event_client.register(Path)
def _catalog_to_client(path):
    """
    Turn a str or directory into a client.

    If a single file, try to read as events, if directory init event bank.
    """
    path = Path(path)  # ensure we are working with a path
    assert path.exists()
    if path.is_dir():
        return get_event_client(obsplus.EventBank(path))
    else:
        return get_event_client(obspy.read_events(str(path)))


def obj_to_dict(obj):
    """
    Return the dict representation of an obspy object.

    Attributes and type are determined from the docstrings of the object.
    """
    try:
        return _TO_DICT_FUNCS[type(obj)](obj)
    except KeyError:
        params = _get_params_from_docs(obj)
        # create function for processing and register
        _TO_DICT_FUNCS[type(obj)] = _getattr_factory(params)
        # register function for future caching
        return _TO_DICT_FUNCS[type(obj)](obj)


def _camel2snake(name):
    """
    Convert CamelCase to snake_case.

    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    return s2


@lru_cache()
def make_class_map():
    """
    Return a dict that maps names in QML to the appropriate obspy class.
    """

    # add "special" cases to mapping
    out = dict(mag_errors=QuantityError)
    out.update({x: obspy.UTCDateTime for x in UTC_KEYS})

    def _add_lower_and_plural(name, cls):
        """ add the lower case and plural case to dict"""
        name_lower = _camel2snake(name)
        name_plural = name_lower + "s"
        out[name_lower] = cls
        out[name_plural] = cls  # add both singular and plural

    # iterate all classes contained in core.event and add to dict
    for name, cls in ev.__dict__.items():
        if not isinstance(cls, type):
            continue
        if hasattr(cls, "_property_dict"):
            for name_, obj_type in cls._property_dict.items():
                # skip enums, object creation handles validation of these
                if isinstance(obj_type, Enum):
                    continue
                _add_lower_and_plural(name_, obj_type)
        _add_lower_and_plural(name, cls)
    return out


def get_preferred(event: Event, what: str, init_empty=False):
    """
    get the preferred object (eg origin, magnitude) from the event.

    If not defined use the last in the list. If list is empty init empty
    object.

    Parameters

    -----------
    event: obspy.core.event.Event
        The instance for which the preferred should be sought.
    what: the preferred item to get
        Can either be "magnitude", "origin", or "focal_mechanism".
    init_empty
        If True, rather than return None when no preferred object is found
        create an empty object of the appropriate class and return it.
    """

    def _none_or_empty():
        """ Return None or an empty object of correct type. """
        if init_empty:
            return getattr(obspy.core.event, what.capitalize())()
        else:
            return None

    pref_type = {
        "magnitude": ev.Magnitude,
        "origin": ev.Origin,
        "focal_mechanism": ev.FocalMechanism,
    }

    prefname = "preferred_" + what
    whats = what + "s"
    obj = getattr(event, prefname)()
    if obj is None:  # get end of list
        pid = getattr(event, prefname + "_id")
        if pid is None:  # no preferred id set, return last in list
            try:  # if there is None return an empty one
                obj = getattr(event, whats)[-1]
            except IndexError:  # object has no whats (eg magnitude)
                # TODO why not return None here?
                return _none_or_empty()
        else:  # there is an id, it has just come detached, try to find it
            potentials = {x.resource_id.id: x for x in getattr(event, whats)}
            if pid.id in potentials:
                obj = potentials[pid.id]
            else:
                var = (pid.id, whats, str(event))
                warnings.warn("cannot find %s in %s for event %s" % var)
                try:
                    obj = getattr(event, whats)[-1]
                except IndexError:  # there are no "whats", return None
                    return _none_or_empty()
    # make sure the correct output type is returned
    cls = pref_type[what]
    assert isinstance(obj, cls), f"{type(obj)} is not a {cls}, wrong type returned"
    return obj
