"""
functions for merging catalogs together
"""

import warnings
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
from obspy.core.event import Catalog, Origin, Event

import obsplus
from obsplus import validate_catalog
from obsplus.utils.events import bump_creation_version


def merge_events(eve1: Event, eve2: Event, reject_old: bool = True) -> Event:
    """
    Merge picks and amplitudes of two events together.

    This function attempts to merge picks and amplitudes of two events
    together that may have different resource_ids in some attributes. The
    second event is imposed on the first which is modified in place.

    Parameters
    ----------
    eve1 : Catalog
        The first (former) event
    eve2 : Catalog
        The second (new) event
    reject_old : bool
        If True, reject anything in eve1 not in eve2

    Returns
    -------
    Event
        The merged events
    """
    _merge_picks(eve1, eve2, reject_old=reject_old)
    _merge_amplitudes(eve1, eve2, reject_old=reject_old)
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


def _merge_picks(eve1, eve2, reject_old=False):
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

    # reject old
    if reject_old:
        _reject_old(eve1.picks, "phase_hint", widp_p2)
        # eve1.picks = [x for x in eve1.picks if _hash_wid(x, "phase_hint") in widp_p2]


def _merge_amplitudes(eve1, eve2, reject_old=False):
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
    # reject old
    if reject_old:
        _reject_old(eve1.amplitudes, "magnitude_hint", pid1_a2)
        # eve1.amplitudes = [x for x in eve1.amplitudes if x.pick_id.id in pid1_a2]


def _reject_old(objs, hash_attr, checklist):
    """Set the evaluation status of outdated objects to 'rejected'"""
    for x in objs:
        try:
            wid = _hash_wid(x, hash_attr)
        except AttributeError:
            # It is not a valid Amplitude... reject it
            x.evaluation_status = "rejected"
        else:
            if wid not in checklist:
                x.evaluation_status = "rejected"


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
    merge_events(old_event, new_event, reject_old=False)
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
    """associate the origin arrivals with correct picks from old event"""
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


def associate_merge(
    event: Event,
    new_catalog: Union[Catalog, Event],
    median_tolerance: float = 1.0,
    reject_old: bool = False,
) -> Event:
    """
    Merge the "closest" event in a catalog into an existing event.

    Finds the closest event in new_catalog to event using median pick
    times, then calls :func:`obsplus.events.merge.merge_events` to merge
    the events together.

    Parameters
    ----------
    event
        The base event which will be modified in place.
    new_catalog
        A new catalog or event which contains picks.
    median_tolerance
        The tolerance, in seconds, of the median pick for associating
        events in new_catalog into event.
    reject_old
        Reject any picks/amplitudes in old event if not found in new
        event.
    """

    def _get_pick_median(time_ser):
        """
        Return a (close enough) approximation of the median for datetimes in ns.
        """
        int_median = int(time_ser.astype(np.int64).median())
        return int_median

    def _get_associated_event_id(new_picks, old_picks):
        """Return the associated event id"""
        new_med = new_picks.groupby("event_id")["time"].apply(_get_pick_median)
        old_med = _get_pick_median(old_picks["time"])
        diffs = abs(new_med - old_med)
        # check on min tolerance, if exceeded return empty
        if diffs.min() / 1_000_000_000 > median_tolerance:
            return None
        return diffs.idxmin()

    # Get list-like of events from new_catalog
    new_cat = new_catalog if isinstance(new_catalog, Catalog) else [new_catalog]
    assert len(new_catalog) > 0
    # Get dataframes of event info
    new_pick_df = obsplus.picks_to_df(new_cat)
    old_pick_df = obsplus.picks_to_df(event)
    eid = _get_associated_event_id(new_pick_df, old_pick_df)
    new_event = {str(x.resource_id): x for x in new_catalog}.get(eid)
    # The association failed, just return original event
    if new_event is None:
        return event
    return merge_events(event, new_event, reject_old=reject_old)


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
