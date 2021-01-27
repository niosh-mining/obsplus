"""
A module defining mappers which extract dataframe from ObsPy's events
structure.
"""
import obsplus.events.schema as es
from obsplus.structures.dfmap import DFMap


class EventBasic(DFMap):
    """
    A mapper to extract only the basic information about events.

    Only the information needed for :meth:`~obsplus.EventBank.get_events`
    is extracted.
    """

    _model = es.Catalog
    event_id = es.Event.resource_id
    event_description = es.Event.event_descriptions[0].text

    # get the preferred origin/magnitudes
    _por = es.Event.preferred_origin_id | es.Event.origins[-1]
    _pmag = es.Event.preferred_magnitude_id | es.Event.magnitudes[-1]

    latitude = _por.latitude
    longitude = _por.longitude
    depth = _por.depth
    time = _por.time
    standard_error = _por.quality.standard_error

    # get magnitude info
    magnitude = _pmag.mag


class Event(EventBasic):
    """A dataframe mapper for extracting more event information."""

    # get origin info
    _por = es.Event.preferred_origin_id | es.Event.origins[0]
    _pmag = es.Event.preferred_magnitude_id | es.Event.magnitudes[0]

    preferred_origin_id = _por
    preferred_magnitude_id = _pmag
    magnitude_type = _por.magnitude_type

    # origin quality info
    associated_phase_count = _por.quality.associated_phase_count
    used_phase_count = _por.quality.used_phase_count
    azimuthal_gap = _por.quality.azmuthial_gap

    _origin_uncert = _por.origin.origin_uncertainty
    horizontal_uncertainty = _origin_uncert.horizontal_uncertainty

    # local_magnitude=float,
    # moment_magnitude=float,
    # duration_magnitude=float,
    # p_phase_count=float,
    # s_phase_count=float,
    # p_pick_count=float,
    # s_pick_count=float,
    # station_count=float,
    # vertical_uncertainty=float,
    # get things from creation info
    _ci = es.Event.creation_info
    author = _ci.author
    agency_id = _ci.agency_id
    creation_time = _ci.creation_time
    updated = _ci
    version = _ci.version


class Picks(DFMap):
    """A dataframe mapper. """

    _model = es.Catalog
    _pick = es.Pick

    resource_id = _pick.resource_id
    time = _pick.time
    seed_id = _pick.waveform_id.seed_string
    filter_id = _pick.filter_id
    method_id = _pick.method_id
    horizontal_slowness = _pick.horizontal_slowness
    backazimuth = _pick.backazimuth
    onset = _pick.onset
    phase_hint = _pick.phase_hint
    polarity = _pick.polarity
    evaluation_mode = _pick.evaluation_mode
