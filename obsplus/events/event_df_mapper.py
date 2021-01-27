"""
Dataframe mappers for events
"""
import obsplus.events.schema as es
import obsplus.structures.mill as mill


class Events:
    """A dataframe mapper for getting event summaries."""

    # get origin info
    _por = es.Event.preferred_origin_id | es.Event.origins[0]
    preferred_origin_id = _por
    latitude = _por.latitude
    longitude = _por.longitude
    depth = _por.depth
    time = _por.time

    # get magnitude info
    _pmag = es.Event.preferred_magnitude_id | es.Event.magnitudes[0]
    perferred_magnitude_id = _pmag
    magnitude = _pmag.mag

    event_description = es.Event.event_descriptions[0].text
    associated_phase_count = es.Event
    azimuthal_gap=float,
    event_id=str,
    horizontal_uncertainty=float,
    local_magnitude=float,
    moment_magnitude=float,
    duration_magnitude=float,
    magnitude_type=str,
    p_phase_count=float,
    s_phase_count=float,
    p_pick_count=float,
    s_pick_count=float,
    standard_error=float,
    used_phase_count=float,
    station_count=float,
    vertical_uncertainty=float,
    # get things from creation info
    _ci = es.Event.creation_info
    author = _ci.author
    agency_id = _ci.agency_id
    creation_time = _ci.creation_time
    updated = _ci
    version = _ci.version



class Picks:
    """A dataframe mapper. """
    time = es.Pick.time


