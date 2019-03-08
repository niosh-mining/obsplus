"""
Functions for validating obspy events objects
"""
from typing import Union, Optional

from obspy.core.event import (
    Catalog,
    Event,
    ResourceIdentifier,
    QuantityError,
    WaveformStreamID,
)

import obsplus
from obsplus.constants import ORIGIN_FLOATS, QUANTITY_ERRORS
from obsplus.utils import yield_obj_parent_attr, replace_null_nlsc_codes

CATALOG_VALIDATORS = []

def _none_or_type(obj, type_check):
    """ return if obj is None or of type type_check """
    if obj is None:
        return True
    else:
        return isinstance(obj, type_check)


def catalog_validator(func):
    """ register a catalog_validator function, which should take a single
    events as the only arguments. If the check fails an exception should
    be raised """
    CATALOG_VALIDATORS.append(func)
    return func


@catalog_validator
def set_preferred_values(event: Event, **kwargs):
    """ set the preferred values to the last in the list if they are not
    defined """
    if not event.preferred_origin_id and len(event.origins):
        event.preferred_origin_id = event.origins[-1].resource_id
    if not event.preferred_magnitude_id and len(event.magnitudes):
        event.preferred_magnitude_id = event.magnitudes[-1].resource_id
    if not event.preferred_focal_mechanism_id and len(event.focal_mechanisms):
        focal_mech_id = event.focal_mechanisms[-1].resource_id
        event.preferred_focal_mechanism_id = focal_mech_id


@catalog_validator
def attach_all_resource_ids(event: Event, **kwargs):
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


@catalog_validator
def check_arrivals_pick_id(event: Event, **kwargs):
    """ check that all arrivals link to a pick object, if they are not
     attached set the referred object attr """
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


@catalog_validator
def check_origins(event: Event, **kwargs):
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


@catalog_validator
def check_picks(event: Event, **kwargs):
    """
    Checks for errors with phase picks on each station

    This function will check for duplicate picks on each station (i.e. more
    than one P or S per station), if there are any S or IAML picks before
    P picks on each station, and if there are more than one IAML pick per 
    channel.
    """
    pdf = obsplus.picks_to_df(event)
    pdf = pdf.loc[pdf.evaluation_status != "rejected"]

    def dup_picks(phase_hint, df=pdf, event_id=event.resource_id.id, 
                  on='station'):
        df = df.loc[df.phase_hint == phase_hint]
        bad = df.loc[df[on].duplicated()][on].tolist()
        assert len(bad) == 0, (
            f'Duplicate {phase_hint} picks found\n'
            f'event_id: {event_id}, '
            f'{on}/s: {bad}')
    
    def pick_order(g, sp, ap, event_id=event.resource_id.id):
        # Check P before S
        temp = {'P','S'}
        if temp.issubset(g.phase_hint):
            p_pick = g.loc[g.phase_hint == "P"].iloc[0]
            s_pick = g.loc[g.phase_hint == "S"].iloc[0]
            if p_pick.time > s_pick.time:
                sp.append(g.name)
        # Check P before IAML
        temp = {'P','IAML'}
        if temp.issubset(g.phase_hint):
            p_pick = g.loc[g.phase_hint == "P"].iloc[0]
            amp_picks = g.loc[g.phase_hint == "IAML"]
            bad = []
            for _, amp in amp_picks.iterrows():
                if p_pick.time > amp.time:
                    bad.append(amp.seed_id)
            ap.extend(bad)
            
    # Checking for duplicated picks
    dup_picks('P')
    dup_picks('S')
    dup_picks('IAML', on='seed_id')
    
    # Checking that picks are in acceptable order
    gb = pdf.groupby("station")
    sp = []
    ap = []
    gb.apply(pick_order, sp, ap)
    assert len(sp) == 0, (
            'S pick found before P pick:\n'
            f'station/s: {sp}')
    assert len(ap) == 0, (
            'IAML pick found before P pick:\n'
            f'seed_id/s: {ap}')
    
    
@catalog_validator
def check_p_lims(event: Event, p_lim=None, **kwargs):
    """
    Check for P picks that aren't within p_lim of the median pick (if provided)
    """
    if p_lim is not None:
        df = obsplus.picks_to_df(event)
        df = df.loc[(df.evaluation_status != "rejected") & 
                    (df.phase_hint == 'P')]
        med = df.time.median()
        bad = df.loc[abs(df.time - med) > p_lim]
        assert len(bad) == 0, (
                'Outlying P pick found:\n'
                f'event_id: {event.resource_id.id}, '
                f'seed_id/s: {bad.seed_id.tolist()}')


@catalog_validator
def check_amp_lims(event: Event, amp_lim=None, **kwargs):
    """
    Check for amplitudes that aren't below amp_lim (if provided)
    """
    if amp_lim is not None:
        bad = []
        for amp in event.amplitudes:
            if amp.generic_amplitude > amp_lim:
                wid = amp.waveform_id
                nslc = (f'{wid.network_code}.{wid.station_code}.'
                        f'{wid.location_code}.{wid.channel_code}')
                bad.append(nslc)            
        assert len(bad) == 0, (
                'Above limit amplitude found:\n'
                f'event_id: {event.resource_id.id}, '
                f'seed_id/s: {bad}')


@catalog_validator
def check_amp_filts(event: Event, filt_amps=None, **kwargs):
    """
    Check that all amplitudes have a specified filter id (if provided)
    """
    if filt_amps is not None:
        if type(filt_amps) is ResourceIdentifier:
            filt_amps = filt_amps.id
        bad = []
        bad_filters = []
        for amp in event.amplitudes:
            if amp.filter_id.id != filt_amps:
                wid = amp.waveform_id
                nslc = (f'{wid.network_code}.{wid.station_code}.'
                        f'{wid.location_code}.{wid.channel_code}')
                bad.append(nslc)  
                if amp.filter_id.id not in bad_filters:
                    bad_filters.append(amp.filter_id.id)
        assert len(bad) == 0, (
                'Unexpected amplitude filter found:\n'
                f'event_id: {event.resource_id.id}, '
                f'seed_id/s: {bad}, '
                f'filters_used: {set(bad_filters)}')   


@catalog_validator
def check_z_amps(event: Event, no_z_amps=False, **kwargs):
    """
    Check for IAML picks on Z channels (if no_z_amps is True)
    """
    if no_z_amps:
        df = obsplus.picks_to_df(event)
        df = df.loc[(df.evaluation_status != "rejected") & 
                    (df.phase_hint == 'IAML')]
        bad = df.loc[df.channel.str.endswith('Z')].seed_id.tolist()
        assert len(bad) == 0, (
                'Amplitude pick on Z axis found:\n'
                f'event_id: {event.resource_id.id}, '
                f'seed_id/s: {bad}')
        
        
@catalog_validator
def check_amp_times(event: Event, **kwargs):
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
            nslc = (f'{wid.network_code}.{wid.station_code}.'
                    f'{wid.location_code}.{wid.channel_code}')
            bad.append(nslc)
    assert len(bad) == 0, (
            'Mismatched amplitude and pick times found:\n'
            f'event_id: {event.resource_id.id}, '
            f'seed_id/s: {bad}, ')      





# register the nullish nslc code replacement
catalog_validator(replace_null_nlsc_codes)


def validate_catalog(events: Union[Catalog, Event], 
                     **kwargs) -> Optional[Union[Catalog, Event]]:
    """
    Perform tchecks on a events or event object.

    This function will try to fix any issues but will raise if it cannot.

    Parameters
    ----------
    events
        The events or event to check

    """
    # TODO: asssert or print bool?

    cat = events if isinstance(events, Catalog) else Catalog(events=[events])
    for event in cat:
        for func in CATALOG_VALIDATORS:
            func(event, **kwargs)
    return events

