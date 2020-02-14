"""
functions for merging catalogs together
"""

import warnings
from collections import OrderedDict
from typing import Optional

from obspy.core.event import Catalog, Origin, Event

from obsplus import validate_catalog
from obsplus.utils.events import bump_creation_version


def merge_events(eve1: Event, eve2: Event, delete_old: bool = True) -> Event:
    """
    Merge picks and amplitudes of two events together.

    This function attempts to merge pciks and amplitudes of two events
    together that may have different resource_ids in some attributes. The
    second event is imposed on the first which is modified in place.

    Parameters
    ----------
    eve1 : Catalog
        The first (former) event
    eve2 : Catalog
        The second (new) event
    delete_old : bool
        If True delete from eve1 anything not in eve2

    Returns
    -------
    Event
        The merged events
    """
    _merge_picks(eve1, eve2, delete_old=delete_old)
    _merge_amplitudes(eve1, eve2, delete_old=delete_old)
    return eve1


def _generate_pick_phase_maps(eve1, eve2):
    maps = {}
    maps["widp_p1"] = _hash_wids(eve1.picks, "phase_hint")
    maps["widp_p2"] = _hash_wids(eve2.picks, "phase_hint")
    maps["widm_a1"] = _hash_wids(eve1.amplitudes, "magnitude_hint")
    maps["widm_a2"] = _hash_wids(eve2.amplitudes, "magnitude_hint")
    maps["id_p1"] = {x.resource_id.id for x in eve1.picks}
    maps["id_p2"] = {x.resource_id.id for x in eve2.picks}
    maps["id1_p2"] = {
        val.resource_id.id: maps["widp_p2"].get(key, None)
        for key, val in maps["widp_p1"].items()
    }
    maps["id2_p1"] = {
        val.resource_id.id: maps["widp_p1"].get(key, None)
        for key, val in maps["widp_p2"].items()
    }
    return maps


def _merge_picks(eve1, eve2, delete_old=False):
    """
    Merge a list of objects that have waveform ids (arrivals, picks,
    amplitudes)
    """
    maps = _generate_pick_phase_maps(eve1, eve2)
    # new attributes that should overwrite old ones
    attrs_no_update = {"resource_id", "force_resource_id"}
    widp_p1 = maps["widp_p1"]
    widp_p2 = maps["widp_p2"]

    # for picks in old and new
    for key in set(widp_p1) & set(widp_p2):
        if widp_p1[key] != widp_p2[key]:
            bump_creation_version(widp_p2[key])  # bump creation info
            pd2 = widp_p2[key].__dict__
            update_vals = {x: pd2[x] for x in pd2 if x not in attrs_no_update}
            widp_p1[key].__dict__.update(update_vals)
    # for new picks append
    for key in set(widp_p2) - set(widp_p1):
        eve1.picks.append(widp_p2[key])

    # delete old
    if delete_old:
        eve1.picks = [x for x in eve1.picks if _hash_wid(x, "phase_hint") in widp_p2]


def _merge_amplitudes(eve1, eve2, delete_old=False):
    """Merge the amplitudes together."""
    attrs_no_update = {"pick_id", "resource_id", "force_resource_id"}
    maps = _generate_pick_phase_maps(eve1, eve2)

    pid1_a1 = {x.pick_id.id: x for x in eve1.amplitudes}
    pid1_a2 = {}  # map all amplitude 2 back to pick ids on first event
    for amp in eve2.amplitudes:
        aid = amp.pick_id
        if aid is not None and maps["id2_p1"][aid.id] is not None:
            key = maps["id2_p1"][aid.id].resource_id.id
            pid1_a2[key] = amp
    # common keys
    for key in set(pid1_a1) & set(pid1_a2):
        amp1, amp2 = pid1_a2[key], pid1_a1[key]
        if amp1 == amp2:
            continue
        bump_creation_version(amp2)  # bump creation info
        pd2 = amp2.__dict__
        update_vals = {x: pd2[x] for x in pd2 if x not in attrs_no_update}
        amp1.__dict__.update(update_vals)
    # for new amplitudes append
    for key in set(pid1_a2) - set(pid1_a1):
        eve1.amplitudes.append(pid1_a2[key])
    # delete old
    if delete_old:
        eve1.amplitudes = [x for x in eve1.amplitudes if x.pick_id.id in pid1_a2]


def attach_new_origin(
    old_event: Event,
    new_event: Event,
    new_origin: Origin,
    preferred: bool,
    index: Optional[int] = None,
) -> Catalog:
    """
    Attach a new origin to an existing events object.

    Parameters
    ----------
    old_event : obspy.core.event.Event
        The old event that will receive the new origin
    new_event : obspy.core.event.Event
        The new event that contains the origin, needed for merging picks
        that may not exist in old_event
    new_origin : obspy.core.event.Origin
        The new origin that will be attached to old_event
    preferred : bool
        If True mark the new origin as the preferred_origin
    index : int or None
        The origin index of old_cat that new_origin will overwrite, if None
        append the new_origin to old_cat.origins

    Returns
    -------
    obspy.Catalog
        modifies old_cat in-place, returns old_catalog
    """
    # make sure all the picks/amplitudes in new_event are also in old_event
    merge_events(old_event, new_event, delete_old=False)
    # point the arrivals in the new origin at the old picks
    _associate_picks(old_event, new_event, new_origin)
    # append the origin
    if index is not None:  # if this origin is to replace another
        try:
            old_ori = old_event.origins[index]
        except IndexError:
            msg = ("%d is not valid for an origin list of length %d") % (
                index,
                len(old_event.origins),
            )
            msg += " appending new origin to end of list"
            warnings.warn(msg)
            old_event.origins.append(new_origin)
        else:
            # set resource id and creation info
            new_origin.resource_id = old_ori.resource_id
            new_origin.creation_info = old_ori.creation_info
            old_event.origins[index] = new_origin
    else:
        old_event.origins.append(new_origin)
    # bump origin creation info
    bump_creation_version(new_origin)
    # set preferred
    if preferred:
        old_event.preferred_origin_id = new_origin.resource_id
    validate_catalog(old_event)
    return old_event


def _associate_picks(old_eve, new_event, new_origin):
    """ associate the origin arrivals with correct picks from old event"""
    picks = old_eve.picks  # picks of old cat_name
    new_resource_pick_dict = {x.resource_id.id: x for x in new_event.picks}
    old_pick_dict = _hash_wids(picks, "phase_hint")
    for arrival in new_origin.arrivals:
        # associate picks together
        new_pick = new_resource_pick_dict[arrival.pick_id.id]
        new_pick_hash = _hash_wid(new_pick, "phase_hint")
        # get corresponding old pick and swap resource id of arrival
        old_pick = old_pick_dict[new_pick_hash]
        arrival.pick_id = old_pick.resource_id


# ---------- silly hash functions for getting around resource_ids (sorta)


def _hash_wids(objs, extra_attr=None):
    out = OrderedDict()
    for obj in objs:
        try:
            key = _hash_wid(obj, extra_attr)
        except AttributeError:  # if cant hash object
            continue
        else:
            out[key] = obj
    return out


def _hash_wid(obj, extra_attr):
    wid = obj.waveform_id
    extra = getattr(obj, extra_attr) if extra_attr else ""
    key = "-".join([wid.network_code, wid.station_code, wid.channel_code, extra])
    return key
