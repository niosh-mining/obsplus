"""
Functions for validating obspy events objects
"""
from typing import Union, Optional

from obsplus.constants import ORIGIN_FLOATS, QUANTITY_ERRORS
from obsplus.utils import yield_obj_parent_attr
import obsplus
from obspy.core.event import Catalog, Event, ResourceIdentifier, QuantityError
from obspy import UTCDateTime

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
def set_preferred_values(event: Event):
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


@catalog_validator
def check_arrivals_pick_id(event: Event):
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
                

def validate_catalog(events: Union[Catalog, Event],) -> Optional[Union[Catalog, Event]]:
    """
    Perform tchecks on a events or event object.

    This function will try to fix any issues but will raise if it cannot.

    Parameters
    ----------
    events
        The events or event to check

    """

    cat = events if isinstance(events, Catalog) else Catalog(events=[events])
    for event in cat:
        for func in CATALOG_VALIDATORS:
            func(event)
    return events


def check_picks(cat: Catalog):
    """ 
    Checks for errors with phase picks
    
    This function will check for duplicate picks on each station (i.e. more 
    than one P or S per station) as well as if there are any S picks before 
    P picks on each station.
    
    Parameters
    ----------
    cat
        Obspy catalog to validate
           
    """
    def fn(df):
        # No duplicates
        assert not any(df.phase_hint.duplicated())
        
        # Check p before s
        if ps.issubset(df.phase_hint):
            p_pick = df.loc[df.phase_hint == 'P'].iloc[0]
            s_pick = df.loc[df.phase_hint == 'S'].iloc[0]
            assert p_pick.time < s_pick.time
            
    ps = {'P', 'S'}
    
    pdf = obsplus.picks_to_df(cat)
    pdf = pdf.loc[(pdf.evaluation_status != 'rejected') & (pdf.phase_hint.isin(ps))]
    gb = pdf.groupby(['event_id', 'station'])
    gb.apply(fn)
        

        