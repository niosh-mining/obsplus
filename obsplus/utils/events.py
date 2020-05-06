"""
General utility functions which are not specific to one data type.
"""

import copy
import logging
import re
import warnings
from contextlib import suppress
from functools import lru_cache, singledispatch
from pathlib import Path
from typing import Optional, Callable, Iterable

import obspy
import obspy.core.event as ev
import pandas as pd
from obspy.core.event import Catalog, Event, ResourceIdentifier, WaveformStreamID
from obspy.core.event.base import QuantityError
from obspy.core.util.obspy_types import Enum

import obsplus
from obsplus.constants import (
    catalog_or_event,
    catalog_component,
    EVENT_ATTRS,
    UTC_KEYS,
    event_clientable_type,
    EVENT_PATH_STRUCTURE,
    EVENT_NAME_STRUCTURE,
)
from obsplus.exceptions import ValidationError
from obsplus.interfaces import EventClient
from obsplus.utils.bank import EVENT_EXT, _get_time_values
from obsplus.utils.misc import yield_obj_parent_attr, _get_path
from obsplus.utils.pd import get_seed_id_series
from obsplus.utils.time import _get_event_origin_time, to_utc


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


def prune_events(events: catalog_or_event) -> Catalog:
    """
    Remove all the unreferenced rejected objects from an event or catalog.

    This function first creates a copy of the event/catalog. Then it looks
    all objects with rejected evaluation status', which are not referred to
    by any un-rejected object, and removes them.

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
        edges = []  # list of tuples (parent, child)
        rejected_rid = set()  # {resource_id, }
        rejected_opa = {}  # {resource_id: [obj, parent, attr]}
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
        # if maybe_obj is obj:
        #     breakpoint()
        #     setattr(parent, attr, None)
        # else:
        assert isinstance(maybe_obj, list)
        maybe_obj.remove(obj)

    # iterate events, find and destroy rejected orphans (that sounds bad...)
    for event in events:
        event = event.copy()
        validate_catalog(event)
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


@singledispatch
def strip_events(
    events: catalog_or_event, reject_evaluation_status: Iterable = "rejected"
) -> catalog_or_event:
    """
    Removes all derivative data and rejected objects from an event or catalog

    This is a nuclear option for when processing goes horribly wrong. It will
    only keep picks and amplitudes that are not rejected in addition to the
    first event description for the event.

    Parameters
    ----------
    events
        The events to strip
    reject_evaluation_status
        Reject picks and amplitudes that have this as an evaluation status
        (accepts either a single value or a list)

    Returns
    -------
    The stripped events
    """
    # Make sure this returns a new catalog
    out = Catalog()
    for eve in events:
        out.append(strip_events(eve, reject_evaluation_status=reject_evaluation_status))
    return out


@strip_events.register(Event)
def _strip_event(eve, reject_evaluation_status="rejected") -> Event:
    """Strip down a single event"""
    # Make sure this returns a copy of the events
    eve = eve.copy()
    # Remove derivative data
    for att in ["origins", "magnitudes", "station_magnitudes", "focal_mechanisms"]:
        setattr(eve, att, [])
    # Unset preferred anything
    for att in ["origin", "magnitude", "focal_mechanism"]:
        setattr(eve, f"preferred_{att}_id", None)
    # Filter the picks
    if isinstance(reject_evaluation_status, str):
        reject_evaluation_status = [reject_evaluation_status]
    eve.picks = [
        p for p in eve.picks if p.evaluation_status not in reject_evaluation_status
    ]
    # Filter the amplitudes
    amps = []
    for amp in eve.amplitudes:
        # Reject if the evaluation status is in the reject list
        if amp.evaluation_status in reject_evaluation_status:
            continue
        if amp.pick_id:
            pick = amp.pick_id.get_referred_object()
            # Reject if the evaluation status of the pick tied to the amplitude is in
            # the reject list
            if pick and pick.evaluation_status in reject_evaluation_status:
                continue
        amps.append(amp)
    eve.amplitudes = amps
    # Filter the event descriptions
    if len(eve.event_descriptions):
        eve.event_descriptions = [eve.event_descriptions[0]]
    return eve


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
    nslc_series = get_seed_id_series(df)
    for event in cat:
        if not event.origins:  # make new origin
            picks = event.picks_to_df()
            picks = picks.loc[
                (~(picks["evaluation_status"] == "rejected"))
                & (picks["phase_hint"].isin(phase_hints))
            ]
            if not len(picks):
                msg = f"{event} has no acceptable picks to create origin"
                raise ValidationError(msg)
            # get first pick, determine time/station used
            first_pick = picks.loc[picks["time"].idxmin()]
            seed_id = first_pick["seed_id"]
            # find channel corresponding to pick
            df_chan = df[nslc_series == seed_id]
            if not len(df_chan):
                raise ValidationError(f"{seed_id} not found in inventory")
            ser = df_chan.iloc[0]
            # create origin
            ori = _create_first_pick_origin(first_pick, ser, depth=depth)
            event.origins.append(ori)
    return events


def _create_first_pick_origin(first_pick, channel_ser, depth):
    """ Create an origin based on first pick and a channel series. """
    msg = (
        "origin fixed to location and time of earliest pick by "
        f"obsplus version {obsplus.__last_version__}"
    )
    comment = ev.Comment(text=msg)
    odict = dict(
        time=to_utc(first_pick["time"]),
        latitude=channel_ser["latitude"],
        longitude=channel_ser["longitude"],
        depth=depth,
        time_fixed=True,
        comments=[comment],
    )
    return ev.Origin(**odict)


def get_seed_id(obj: catalog_component) -> str:
    """
    Get the NSLC associated with an station-specific object.

    Parameters
    ----------
    obj
        The object for which to retrieve the seed id. Can be anything that
        has a waveform_id attribute or refers to an object with a
        waveform_id attribute.

    Returns
    -------
    str :
        The seed_id in the form of "network.station.location.channel"
    """
    if isinstance(obj, WaveformStreamID):
        # Get the seed id
        return obj.get_seed_string()
    if isinstance(obj, ResourceIdentifier):
        # Get the next nested object
        return get_seed_id(obj.get_referred_object())
    # The order of this list matters! It should first try waveform_id and then
    # go from the shallowest nested attribute to the deepest
    attrs = ["waveform_id", "station_magnitude_id", "amplitude_id", "pick_id"]
    # Make sure the object has at least one of the required attributes
    if not len(set(attrs).intersection(set(vars(obj)))):
        raise TypeError(f"cannot retrieve seed id for objects of type {type(obj)}")
    # Loop over each of the attributes, if it exists and is not None,
    # go down a level until it finds a seed id
    for att in attrs:
        val = getattr(obj, att, None)
        if val:
            with suppress((TypeError, AttributeError)):
                return get_seed_id(val)
    # If it makes it this far, it could not find a non-None attribute
    # raise assertion error so this still works in validators
    assert 0, f"Unable to fetch a seed id for {obj.resource_id}"


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


@get_event_client.register(ev.Event)
def _event_to_catalog(event):
    return get_event_client(obspy.Catalog(events=[event]))


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
    ----------
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
            obj = potentials.get(pid.id, None)
            if obj is None:  # it wasn't found in the potentials.
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


def _summarize_event(
    event: ev.Event,
    path: Optional[str] = None,
    name: Optional[str] = None,
    path_struct: Optional[str] = None,
    name_struct: Optional[str] = None,
) -> dict:
    """
    Function to extract info from events for indexing.

    Parameters
    ----------
    event
        The event object
    path
        Other Parameters to the file
    name
        Name of the file
    path_struct
        directory structure to create
    name_struct
    """
    res_id = str(event.resource_id)
    out = {
        "ext": EVENT_EXT,
        "event_id": res_id,
        "event_id_short": res_id[-5:],
        "event_id_end": res_id.split("/")[-1],
    }
    t1 = _get_event_origin_time(event)
    out.update(_get_time_values(t1))
    path_struct = path_struct if path_struct is not None else EVENT_PATH_STRUCTURE
    name_struct = name_struct or EVENT_NAME_STRUCTURE
    out.update(_get_path(out, path, name, path_struct, name_struct))
    return out
