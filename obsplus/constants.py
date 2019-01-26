"""
Constants used throughout obsplus
"""
import concurrent.futures
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Union, Optional, Mapping, Any, List, Tuple, TypeVar
from types import MappingProxyType as MapProx

import numpy as np
import obspy
import pandas as pd
import xarray as xr
from obspy import Stream, Trace, UTCDateTime, Inventory, Catalog
from obspy.core.event import Event

from obsplus.interfaces import EventClient, WaveformClient

SKIP_OBS = {
    "creation_info",
    "composite_times",
    "time",
    "creation_time",
    "reference",
    "arrivals",
}

CREATION_KEYS = {"version", "creation_time"}

# names of preferred objects (in obspy Catalog)
PREFERRED = {
    "preferred_magnitude_id",
    "preferred_origin_id",
    "preferred_focal_mechanism_id",
}

# columns required for station data
STATION_DTYPES = MapProx(
    OrderedDict(
        network=str,
        station=str,
        location=str,
        channel=str,
        seed_id=str,
        latitude=float,
        longitude=float,
        elevation=float,
        start_date=float,
        end_date=float,
    )
)

STATION_COLUMNS = tuple(STATION_DTYPES)

# columns required for event_data
EVENT_DTYPES = MapProx(
    OrderedDict(
        time=float,
        latitude=float,
        longitude=float,
        depth=float,
        magnitude=float,
        event_description=str,
        associated_phase_count=int,
        azimuthal_gap=float,
        event_id=str,
        horizontal_uncertainty=float,
        local_magnitude=float,
        moment_magnitude=float,
        duration_magnitude=float,
        magnitude_type=str,
        p_phase_count=int,
        s_phase_count=int,
        p_pick_count=int,
        s_pick_count=int,
        standard_error=float,
        used_phase_count=int,
        stations=float,
        station_count=int,
        vertical_uncertainty=float,
        updated=float,
        author=str,
        agency_id=str,
        creation_time=float,
        version=str,
    )
)

EVENT_COLUMNS = tuple(EVENT_DTYPES)

# columns required for picks
PICK_COLUMNS = (
    "resource_id",
    "event_id",
    "event_time",
    "phase_hint",
    "onset",
    "polarity",
    "time",
    "network",
    "station",
    "location",
    "channel",
)

# keys used to identify UTC objects
UTC_KEYS = ("creation_time", "time", "reference")

# keys to pop out of a json object
JSON_KEYS_TO_POP = {"_id", "_summary"}

# seed id components
NSLC = ("network", "station", "location", "channel")

# the expected dimensions of the standard waveform array
DIMS = ("stream_id", "seed_id", "time")

# ------------------- type aliases (aliai?)

# The waveforms processor type
stream_proc_type = Callable[[Stream], Stream]

# The obspy types for waveform data
wave_type = Union[Stream, Trace, xr.DataArray]

# waveform request type (args for get_waveforms)
waveform_request_type = Tuple[str, str, str, str, UTCDateTime, UTCDateTime]

# the signature of obspy fdsn client
wfcli_type = Callable[[str, str, str, str, UTCDateTime, UTCDateTime], Stream]
waveform_clientable_type = Union[WaveformClient, str, Path, Trace, Stream]

# types accepted by DataFetcher for event info
event_type = Union[Catalog, pd.DataFrame]

# types from which and event client can be created
event_clientable_type = Union[Path, str, Catalog, Event, EventClient]

# a events or event type var
catalog_or_event = TypeVar("catalog_or_event", Catalog, Event)

# types accepted by DataFetcher for stations info
inventory_type = Union[Inventory, pd.DataFrame]

# stations or events
inventory_or_event = Union[Inventory, pd.DataFrame, Catalog, Event]

# types that can be a station client
station_clientable_type = Union[str, Path, Inventory]

# types accepted by DataFetcher
fetch_type = Union[wfcli_type, str]

# time type (anything that can be fed to UTCDateTime)
utc_time_type = Union[UTCDateTime, str, float]

# types for specifying starttimes
starttime_type = Optional[Union[UTCDateTime, Mapping[Any, UTCDateTime]]]

# types for specifying duration
duration_type = Optional[Union[float, Mapping[Any, float]]]

# types that can be used to indicate when an event waveform should start
event_time_type = Union[UTCDateTime, Catalog, Event, float]

# availability output type (return from obspy earthworm client availability)
availability_type = List[Tuple[str, str, str, str, UTCDateTime, UTCDateTime]]

# xarray types
xr_type = Union[xr.DataArray, xr.Dataset]

# basic types
basic_types = Optional[Union[int, float, str, bool]]

# -------------------------- events validation constants

# null quantities for nslc codes
NULL_NSLC_CODES = (None, "--", "None", "nan", "null", np.nan)

# parts of the origin that should have float values
ORIGIN_FLOATS = {"latitude", "longitude", "depth"}

# attributes that constitute errors
QUANTITY_ERRORS = {"depth_errors", "latitude_errors", "longitude_errors", "time_errors"}

# resource_ids that are linked to other resource ids
LINKED_RESOURCE_IDS = {
    "resource_id",
    "pick_id",
    "station_magnitude_id",
    "amplitude_id",
    "origin_id",
    "triggering_origin_id",
}

# formats for parts of the UTC strs
UTC_FORMATS = {
    "year": "%04d",
    "month": "%02d",
    "day": "%02d",
    "hour": "%02d",
    "minute": "%02d",
    "second": "%02d",
}

# input args to UTCDateTime object
UTC_ATTRS = ("year", "month", "day", "hour", "minute", "second", "microsecond")

# input args to Event object
EVENT_ATTRS = (
    "resource_id",
    "force_resource_id",
    "event_type",
    "event_type_certainty",
    "creation_info",
    "event_descriptions",
    "comments",
    "picks",
    "amplitudes",
    "focal_mechanisms",
    "origins",
    "magnitudes",
    "station_magnitudes",
    "preferred_origin_id",
    "preferred_magnitude_id",
    "preferred_focal_mechanism_id",
)

# ------------------------ wavefetcher/bank stuff

# the attrs that can be overwritten temporarily
WAVEFETCHER_OVERRIDES = {"events", "stations", "picks"}

# The default path structure for streams
WAVEFORM_STRUCTURE = "waveforms/{year}/{month}/{day}/{network}/{station}/{channel}"

# The default path structure for events
EVENT_PATH_STRUCTURE = "{year}/{month}/{day}"

# The default name for waveform files
WAVEFORM_NAME_STRUCTURE = "{time}"

# The default name for event files
EVENT_NAME_STRUCTURE = "{time}_{event_id_short}"

# The default path structure for Event streams saved in the get_data module
EVENT_WAVEFORM_PATH_STRUCTURE = "event_waveforms/{year}/{julday}"

TIME_VALUES = [
    "year",
    "month",
    "day",
    "julday",
    "hour",
    "minute",
    "second",
    "microsecond",
    "time",
]

client_type = Union[str, concurrent.futures.Executor]
TIME_PRECISION = obspy.UTCDateTime.DEFAULT_PRECISION
AGG_LEVEL_MAP = dict(network=1, station=2, location=3, channel=4, all=5)

# ------------- Docstring chunks

# the get_waveforms ordered params.
get_waveforms_parameters = """
network : str
    The network code
station : str
    The station code
location : str
    The location code
channel : str
    The channel code
starttime : float or obspy.UTCDateTime
    The desired starttime of the waveforms
endtime : float or obspy.UTCDateTime
    The desired endtime of the waveforms"""

# The subset of the get_events parameters obsplus currently supports.
get_events_parameters = """
starttime: obspy.UTCDateTime or valid input to such, optional
    Limit to events on or after the specified start time.
endtime: obspy.UTCDateTime or valid input to such, optional
    Limit to events on or before the specified end time.
minlatitude: float, optional
    Limit to events with a latitude larger than the specified minimum.
maxlatitude: float, optional
    Limit to events with a latitude smaller than the specified maximum.
minlongitude: float, optional
    Limit to events with a longitude larger than the specified minimum.
maxlongitude: float, optional
    Limit to events with a longitude smaller than the specified maximum.
mindepth: float, optional
    Limit to events with depth, in kilometers, larger than the specified
    minimum.
maxdepth: float, optional
    Limit to events with depth, in kilometers, smaller than the specified
    maximum.
minmagnitude: float, optional
    Limit to events with a magnitude larger than the specified minimum.
maxmagnitude: float, optional
    Limit to events with a magnitude smaller than the specified maximum.
magnitudetype: str, optional
    Specify a magnitude type to use for testing the minimum and maximum
    limits.
eventid: str (or sequence of such), optional
    Select specific event(s) by ID.
limit: int, optional
    Limit the results to the specified number of events.
offset: int, optional
    Return results starting at the event count specified, starting at 1.
contributor: str, optional
    Limit to events contributed by a specified contributor.
updatedafter: obspy.UTCDateTime or valid input to such, optional
    Limit to events updated after the specified time."""
