"""
Functions for validating events according to the obsplus flavor.
"""
from typing import Union, Optional, Collection

import pandas as pd
from obspy.core.event import Catalog, Event, ResourceIdentifier, QuantityError

import obsplus
from obsplus.constants import ORIGIN_FLOATS, QUANTITY_ERRORS, NSLC
from obsplus.utils import get_seed_id_series
from obsplus.utils.misc import yield_obj_parent_attr, iterate, replace_null_nlsc_codes
from obsplus.utils.validate import validator, validate

CATALOG_VALIDATORS = []


def _none_or_type(obj, type_check):
    """ return if obj is None or of type type_check """
    if obj is None:
        return True
    else:
        return isinstance(obj, type_check)


@validator("obsplus", Event)
def set_preferred_values(event: Event):
    """
    Validator to set the preferred values to the last in the list if they are
    not defined.
    """
    if not event.preferred_origin_id and len(event.origins):
        event.preferred_origin_id = event.origins[-1].resource_id
    if not event.preferred_magnitude_id and len(event.magnitudes):
        event.preferred_magnitude_id = event.magnitudes[-1].resource_id
    if not event.preferred_focal_mechanism_id and len(event.focal_mechanisms):
        focal_mech_id = event.focal_mechanisms[-1].resource_id
        event.preferred_focal_mechanism_id = focal_mech_id


@validator("obsplus", Event)
def attach_all_resource_ids(event: Event):
    """ recurse all objects in a events and set referred objects """
    rid_to_object = {}
    # first pass, bind all resource ids to parent
    for rid, parent, attr in yield_obj_parent_attr(event, ResourceIdentifier):
        if attr == "resource_id":
            # if the object has already been set and is not unique, raise
            rid.set_referred_object(parent)
            if rid.id in rid_to_object:
                assert rid.get_referred_object() is rid_to_object[rid.id]
            # else set referred object
            rid_to_object[rid.id] = parent
    # second pass, bind all other resource ids to correct resource ids
    for rid, parent, attr in yield_obj_parent_attr(event, ResourceIdentifier):
        if attr != "resource_id" and rid.id in rid_to_object:
            rid.set_referred_object(rid_to_object[rid.id])


@validator("obsplus", Event)
def check_arrivals_pick_id(event: Event):
    """
    Check that all arrivals link to a pick object, if they are not
    attached set the referred object attr.
    """
    pick_dict = {x.resource_id.id: x for x in event.picks}
    for pick in event.picks:
        # make sure pick has wf_id and phase hint
        assert pick.waveform_id is not None
        assert pick.phase_hint is not None
    for origin in event.origins:
        for arrival in origin.arrivals:
            rid = arrival.pick_id
            assert rid is not None
            assert rid.id in pick_dict


@validator("obsplus", Event)
def check_origins(event: Event):
    """ check the origins and types """
    for ori in event.origins:
        for atr in ORIGIN_FLOATS:
            assert _none_or_type(getattr(ori, atr), float)
        # check depth errors
        for atr in QUANTITY_ERRORS:
            at = getattr(ori, atr)
            assert _none_or_type(at, QuantityError)
            if at is not None:
                assert _none_or_type(at.uncertainty, float)
                assert _none_or_type(at.lower_uncertainty, float)
                assert _none_or_type(at.upper_uncertainty, float)
                assert _none_or_type(at.confidence_level, float)


# register the nullish nslc code replacement
validator("obsplus", Event)(replace_null_nlsc_codes)


@validator("obsplus", Event)
def check_duplicate_picks(event: Event):
    """
    Ensure there are no picks with the same phases on the same channels.
    """

    def dup_picks(df, phase_hint, subset):
        """ Function for checking for duplications. """
        seed_id = get_seed_id_series(df, subset=subset)
        bad = seed_id[seed_id.duplicated()].tolist()
        assert len(bad) == 0, (
            f"Duplicate {phase_hint} picks found\n" f"event_id: {event_id}, "
        )

    # A dict of {phase: column that cant be duplicated}
    phase_duplicates = {"IAML": None, "AML": None}
    # first get dataframe of picks, filter out rejected
    pdf = obsplus.picks_to_df(event)
    pdf = pdf.loc[pdf.evaluation_status != "rejected"]
    # add column for network.station.location
    event_id = str(event.resource_id)
    for phase_hint, sub_df in pdf.groupby("phase_hint"):
        # default to comparing network, station, location
        subset = phase_duplicates.get(phase_hint, NSLC[:-1])
        dup_picks(sub_df, phase_hint, subset=subset)


@validator("obsplus", Event)
def check_pick_order(event: Event,):
    """
    Ensure:
        1. There are no S picks before P picks on any station
        2. There are no amplitude picks before P picks on any station
    """

    def pick_order(g, sp, ap):
        # get sub dfs with phases of interest
        p_picks = g[g["phase_hint"].str.upper() == "P"]
        s_picks = g[g["phase_hint"].str.upper() == "S"]
        amp_picks = g[g["phase_hint"].str.endswith("AML")]
        # there should be one P/S pick
        assert len(p_picks) <= 1 and len(s_picks) <= 1
        # first check that P is less than S, if not append to name of bad
        if len(p_picks) and len(s_picks):
            stime, ptime = s_picks.iloc[0]["time"], p_picks.iloc[0]["time"]
            if (stime < ptime) and not (pd.isnull(ptime) | pd.isnull(stime)):
                sp.append(g.name)
        # next check all amplitude picks are after P
        if len(p_picks) and len(amp_picks):
            ptime = p_picks.iloc[0]["time"]
            bad_amp_picks = amp_picks[amp_picks["time"] < ptime]
            ap.extend(list(bad_amp_picks["seed_id"]))

    # get dataframe of picks, filter out rejected
    pdf = obsplus.picks_to_df(event)
    pdf = pdf.loc[pdf.evaluation_status != "rejected"]
    # get series of network, station
    ns = get_seed_id_series(pdf, subset=NSLC[:3])
    # Checking that picks are in acceptable order
    gb, sp, ap = pdf.groupby(ns), [], []
    gb.apply(pick_order, sp, ap)
    assert len(sp) == 0, "S pick found before P pick:\n" f"station/s: {sp}"
    assert len(ap) == 0, "amplitude pick found before P pick:\n" f"seed_id/s: {ap}"


@validator("obsplus", Event)
def check_p_lims(event: Event, p_lim=None):
    """
    Check for P picks that aren't within p_lim of the median pick (if provided)
    """
    if p_lim is not None:
        df = obsplus.picks_to_df(event)
        df = df.loc[(df.evaluation_status != "rejected") & (df.phase_hint == "P")]
        med = df.time.median()
        bad = df.loc[abs(df.time - med) > p_lim]
        assert len(bad) == 0, (
            "Outlying P pick found:\n"
            f"event_id: {event.resource_id.id}, "
            f"seed_id/s: {bad.seed_id.tolist()}"
        )


@validator("obsplus", Event)
def check_amp_lims(event: Event, amp_lim=None):
    """
    Check for amplitudes that aren't below amp_lim (if provided).
    """
    if amp_lim is not None:
        bad = []
        for amp in event.amplitudes:
            if amp.generic_amplitude > amp_lim:
                wid = amp.waveform_id
                nslc = (
                    f"{wid.network_code}.{wid.station_code}."
                    f"{wid.location_code}.{wid.channel_code}"
                )
                bad.append(nslc)
        assert len(bad) == 0, (
            "Above limit amplitude found:\n"
            f"event_id: {str(event.resource_id)}, "
            f"seed_id/s: {bad}"
        )


@validator("obsplus", Event)
def check_amp_filter_ids(
    event: Event, filter_ids: Optional[Union[str, Collection[str]]] = None
):
    """
    Check that all amplitudes have codes in filter_ids.
    """
    filter_ids = set(str(x) for x in iterate(filter_ids))
    # There is no amplitude specified
    if not filter_ids:
        return
    bad = []
    bad_filters = []
    for amp in event.amplitudes:
        if str(amp.filter_id) not in filter_ids:
            wid = amp.waveform_id
            nslc = (
                f"{wid.network_code}.{wid.station_code}."
                f"{wid.location_code}.{wid.channel_code}"
            )
            bad.append(nslc)
            if amp.filter_id.id not in bad_filters:
                bad_filters.append(amp.filter_id.id)
    assert len(bad) == 0, (
        "Unexpected amplitude filter found:\n"
        f"event_id: {str(event.resource_id)}, "
        f"seed_id/s: {bad}, "
        f"filters_used: {set(bad_filters)}"
    )


@validator("obsplus", Event)
def check_amps_on_z_component(
    event: Event, no_z_amps=False, phase_hints=("AML", "IAML")
):
    """
    Check for amplitude picks on Z channels (if no_z_amps is True).
    """
    if not no_z_amps:
        return
    df = obsplus.picks_to_df(event)
    con1 = df.evaluation_status != "rejected"
    con2 = df.phase_hint.isin(phase_hints)
    con3 = df["channel"].str.endswith("Z")
    _df = df.loc[con1 & con2 & con3]
    assert len(df) == 0, (
        "Amplitude pick on Z axis found:\n"
        f"event_id: {str(event.resource_id)}, "
        f"seed_id/s: {_df['seed_id'].tolist()}"
    )


@validator("obsplus", Event)
def check_amp_times_contain_pick_time(event: Event):
    """
    Check for amplitudes times that don't match the referenced pick time
    """
    bad = []
    for amp in event.amplitudes:
        if amp.time_window is None:
            continue
        amp_t = amp.time_window.reference
        pick = amp.pick_id.get_referred_object()
        if (amp_t is None) or (amp_t != pick.time):
            wid = amp.waveform_id
            nslc = (
                f"{wid.network_code}.{wid.station_code}."
                f"{wid.location_code}.{wid.channel_code}"
            )
            bad.append(nslc)
    assert len(bad) == 0, (
        "Mismatched amplitude and pick times found:\n"
        f"event_id: {event.resource_id.id}, "
        f"seed_id/s: {bad}, "
    )


def validate_catalog(events: Union[Catalog, Event], **kwargs) -> Union[Catalog, Event]:
    """
    Perform checks on a events or event object.

    This function will try to fix any issues but will raise if it cannot. It
    is a simple wrapper around obsplus.validate.validate for the obsplus
    namespace.

    Parameters
    ----------
    events
        The events or event to check
    """
    cat = events if isinstance(events, Catalog) else Catalog(events=[events])
    validate(cat, "obsplus", **kwargs)
    return events
