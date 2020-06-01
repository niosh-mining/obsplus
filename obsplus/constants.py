"""Constants used throughout obsplus."""
from collections import OrderedDict
from os import cpu_count
from pathlib import Path
from types import MappingProxyType as MapProxy
from typing import (
    Callable,
    Union,
    Mapping,
    List,
    Tuple,
    TypeVar,
    MutableSequence,
    Iterable,
)


import numpy as np
import obspy
import pandas as pd
from obspy import Stream, Trace, UTCDateTime, Inventory
from obspy.core.event import Event, Catalog
from obspy.core.util import AttribDict

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

# ----- Extractor constants

# Mapping numpy time types to their internal representation
_DATETIME_TYPE_MAP = {"datetime64[ns]": np.int64, "timedelta64[ns]": np.int64}

# columns required for station data
STATION_DTYPES = OrderedDict(
    network=str,
    station=str,
    location=str,
    channel=str,
    seed_id=str,
    latitude=float,
    longitude=float,
    elevation=float,
    depth=float,
    azimuth=float,
    dip=float,
    sample_rate=float,
    start_date="datetime64[ns]",
    end_date="datetime64[ns]",
)

STATION_COLUMNS = tuple(STATION_DTYPES)

DF_TO_INV_DTYPES = OrderedDict(
    network=str,
    station=str,
    location="location_code",
    channel=str,
    latitude=float,
    longitude=float,
    elevation=float,
    depth=float,
    sample_rate=float,
    start_date=float,
    end_date=float,
)

DF_TO_INV_COLUMNS = tuple(DF_TO_INV_DTYPES)

# columns required for event_data
EVENT_DTYPES = OrderedDict(
    time="datetime64[ns]",
    latitude=float,
    longitude=float,
    depth=float,
    magnitude=float,
    event_description=str,
    associated_phase_count=float,
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
    updated="datetime64[ns]",
    author=str,
    agency_id=str,
    creation_time="datetime64[ns]",
    version=str,
)

# required columns for event dataframe
EVENT_COLUMNS = list(EVENT_DTYPES)

# Event types which are returned from EventBank
EVENT_TYPES_OUTPUT = dict(EVENT_DTYPES)
EVENT_TYPES_OUTPUT.pop("stations", None)
EVENT_TYPES_OUTPUT["path"] = str

# input types for EventBank
INPUT_MAP = {"datetime64[ns]": np.int64}
EVENT_TYPES_INPUT = {i: INPUT_MAP.get(v, v) for i, v in EVENT_TYPES_OUTPUT.items()}

# columns required for picks
PICK_DTYPES = OrderedDict(
    resource_id=str,
    time="datetime64[ns]",
    seed_id=str,
    filter_id=str,
    method_id=str,
    horizontal_slowness=float,
    backazimuth=float,
    onset=str,
    phase_hint=str,
    polarity=str,
    evaluation_mode=str,
    event_time="datetime64[ns]",
    evaluation_status=str,
    creation_time="datetime64[ns]",
    author=str,
    agency_id=str,
    event_id=str,
    network=str,
    station=str,
    location="location_code",
    channel=str,
    uncertainty=float,
    lower_uncertainty=float,
    upper_uncertainty=float,
    confidence_level=float,
)

PICK_COLUMNS = tuple(PICK_DTYPES)

# columns for distance dataframe (output)
DISTANCE_COLUMN_DTYPES = OrderedDict(
    distance_m=float,
    azimuth=float,
    back_azimuth=float,
    distance_degrees=float,
    vertical_distance_m=float,
)

# Columns for dataframe inputs
DISTANCE_COLUMN_INPUT_DTYPES = OrderedDict(
    latitude=float, longitude=float, elevation=float
)

# DISTANCE_COLUMNS = tuple(DISTANCE_COLUMN_DTYPES)
# DISTANCE_INPUT_COLUMNS = tuple(DISTANCE_COLUMN_INPUT_DTYPES)

# columns required for amplitudes
AMPLITUDE_DTYPES = OrderedDict(
    resource_id=str,
    generic_amplitude=float,
    seed_id=str,
    type=str,
    category=str,
    unit=str,
    magnitude_hint=str,
    filter_id=str,
    method_id=str,
    period=float,
    snr=float,
    pick_id=str,
    reference="datetime64[ns]",
    time_begin=float,
    time_end=float,
    scaling_time=float,
    evaluation_mode=str,
    evaluation_status=str,
    creation_time="datetime64[ns]",
    author=str,
    agency_id=str,
    event_time="datetime64[ns]",
    event_id=str,
    network=str,
    station=str,
    location="location_code",
    channel=str,
    uncertainty=float,
    lower_uncertainty=float,
    upper_uncertainty=float,
    confidence_level=float,
)

AMPLITUDE_COLUMNS = (
    "resource_id",
    "event_id",
    "event_time",
    "generic_amplitude",
    "type",
    "magnitude_hint",
    "network",
    "station",
    "location",
    "channel",
)

# columns required for station magnitudes
STATION_MAGNITUDE_DTYPES = OrderedDict(
    resource_id=str,
    mag=float,
    seed_id=str,
    station_magnitude_type=str,
    amplitude_id=str,
    magnitude_id=str,
    origin_id=str,
    method_id=str,
    creation_time="datetime64[ns]",
    author=str,
    agency_id=str,
    event_id=str,
    event_time="datetime64[ns]",
    network=str,
    station=str,
    location="location_code",
    channel=str,
    uncertainty=float,
    lower_uncertainty=float,
    upper_uncertainty=float,
    confidence_level=float,
)

STATION_MAGNITUDE_COLUMNS = (
    "resource_id",
    "mag",
    "station_magnitude_type",
    "network",
    "station",
    "location",
    "channel",
)

# columns required for magnitudes
MAGNITUDE_DTYPES = OrderedDict(
    resource_id=str,
    mag=float,
    seed_id=str,
    magnitude_type=str,
    origin_id=str,
    method_id=str,
    station_count=float,
    azimuthal_gap=float,
    evaluation_mode=str,
    evaluation_status=str,
    creation_time="datetime64[ns]",
    author=str,
    agency_id=str,
    event_id=str,
    event_time="datetime64[ns]",
    uncertainty=float,
    lower_uncertainty=float,
    upper_uncertainty=float,
    confidence_level=float,
)

MAGNITUDE_COLUMNS = ("resource_id", "event_id", "event_time", "mag", "magnitude_type")

# columns required for arrivals
ARRIVAL_DTYPES = OrderedDict(
    resource_id=str,
    seed_id=str,
    pick_id=str,
    phase=str,
    time_correction=float,
    azimuth=float,
    distance=float,
    takeoff_angle=float,
    time_residual=float,
    horizontal_slowness_residual=float,
    backazimuth_residual=float,
    time_weight=float,
    horizontal_slowness_weight=float,
    backazimuth_weight=float,
    earth_model_id=str,
    creation_time="datetime64[ns]",
    author=str,
    agency_id=str,
    network=str,
    station=str,
    location="location_code",
    channel=str,
    origin_id=str,
    origin_time="datetime64[ns]",
)

ARRIVAL_COLUMNS = (
    "resource_id",
    "origin_id",
    "origin_time",
    "pick_id",
    "phase",
    "time_residual",
    "azimuth",
    "distance",
    "time_weight",
    "time_correction",
    "network",
    "station",
    "location",
    "channel",
)

# Waveform datatypes
WAVEFORM_DTYPES = OrderedDict(
    network=str,
    station=str,
    location=str,
    channel=str,
    starttime="datetime64[ns]",
    endtime="datetime64[ns]",
    sampling_period="timedelta64[ns]",
)

# The datatypes needed for putting waveform info into HDF5
WAVEFORM_DTYPES_INPUT = MapProxy(
    {i: _DATETIME_TYPE_MAP.get(v, v) for i, v in WAVEFORM_DTYPES.items()}
)

# keys used to identify UTC objects
UTC_KEYS = ("creation_time", "time", "reference")

# keys to pop out of a json object
JSON_KEYS_TO_POP = {"_id", "_summary"}

# seed id components
NSLC = ("network", "station", "location", "channel")

# the expected dimensions of the standard waveform array
DIMS = ("stream_id", "seed_id", "time")

# Small and BIG UTCDateTimes
BIG_UTC = obspy.UTCDateTime("3000-01-01")
SMALL_UTC = obspy.UTCDateTime("1970-01-01")

# The smallest value an int64 can rep. (used as NaT by datetime64)
MININT64 = np.iinfo(np.int64).min

# The largest value an int64 can rep
MAXINT64 = np.iinfo(np.int64).max

# Large and small np.datetime64[ns] (used when defaults are needed)
SMALLDT64 = np.datetime64(MININT64 + 5_000_000_000, "ns")
LARGEDT64 = np.datetime64(MAXINT64 - 5_000_000_000, "ns")

# The default time value indicating missing values
DEFAULT_TIME = pd.NaT

# an empty time delta to rep. no time distance at all
EMPTYTD64 = np.timedelta64(0, "s")

# path to where obsplus datasets are stored by default
OPSDATA_PATH = Path().home() / "opsdata"

# Number of cores
CPU_COUNT = cpu_count() or 4  # fallback to four is None is returned

# ------------------- type aliases (aliai?)

# Path types
path_types = Union[str, Path]

# number types
number_type = Union[float, int, np.float, np.int, np.complex]

# The waveforms processor type
stream_proc_type = Callable[[Stream], Stream]

# The obspy types for waveform data
wave_type = Union[Stream, Trace]

# Type can can be turned into a UTCDateTime
utc_able_type = Union[str, UTCDateTime, float, np.datetime64, pd.Timestamp]

# waveform request type (args for get_waveforms)
waveform_request_type = Tuple[str, str, str, str, utc_able_type, utc_able_type]

# the signature of obspy fdsn client
wfcli_type = Callable[[str, str, str, str, UTCDateTime, UTCDateTime], Stream]
waveform_clientable_type = Union[WaveformClient, str, Path, Trace, Stream]

# types accepted by DataFetcher for event info
event_type = Union[Catalog, pd.DataFrame, Event]

# types from which and event client can be created
event_clientable_type = Union[Path, str, Catalog, Event, EventClient]

# a events or event type var
catalog_or_event = TypeVar("catalog_or_event", Catalog, Event)

# trace container (Stream, or any mutable collection)
trace_sequence = TypeVar("trace_sequence", Stream, MutableSequence[Trace])

# a component of a catalog object
catalog_component = AttribDict

# types accepted by DataFetcher for stations info
inventory_type = Union[Inventory, pd.DataFrame]

# types that can be a station client
station_clientable_type = Union[str, Path, Inventory]

# types accepted by DataFetcher
fetch_type = Union[wfcli_type, str]

# time type (anything that can be fed to UTCDateTime)
utc_time_type = Union[UTCDateTime, str, float, np.datetime64, pd.Timestamp]

# types that can be used to indicate when an event waveform should start
event_time_type = Union[UTCDateTime, Catalog, Event, float]

# availability output type (return from obspy earthworm client availability)
availability_type = List[Tuple[str, str, str, str, UTCDateTime, UTCDateTime]]

# series to series or ndarray func
series_func_type = Callable[[pd.Series], Union[pd.Series, np.ndarray]]

# type for mapping of functions to apply over callables
column_function_map_type = Mapping[str, series_func_type]

# subpaths type
bank_subpaths_type = Union[path_types, Iterable[path_types]]

# types for bulk waveform requests
bulk_waveform_arg_type = List[Tuple[str, str, str, str, UTCDateTime, UTCDateTime]]

# types which can be used to slice a numpy array
slice_types = Union[int, slice, List[int], Tuple[int, ...]]

# types that indicate time
pd_time_types = (pd.Timestamp, np.datetime64)

# types used to represent an absolute point in time.
absolute_time_types = Union[UTCDateTime, pd.Timestamp, np.datetime64]

# types used to represent relative time
relative_time_types = Union[np.timedelta64, int, float]

# combine the two
time_types = Union[absolute_time_types, relative_time_types]

# time types accepted by trim
trim_time_types = Union[time_types, np.ndarray]

# -------------------------- events validation constants

# null quantities for nslc codes
NULL_SEED_CODES = (None, "--", "None", "nan", "null", np.nan)

# parts of the origin that should have float values
ORIGIN_FLOATS = {"latitude", "longitude", "depth"}

# attributes that constitute errors
QUANTITY_ERRORS = {"depth_errors", "latitude_errors", "longitude_errors", "time_errors"}

# columns needed for bulk waveform request
BULK_WAVEFORM_COLUMNS = tuple(list(NSLC) + ["starttime", "endtime"])

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

# magnitude column names and magnitude type used in event dataframes
MAGNITUDE_COLUMN_TYPES = {
    "moment_magnitude": "MW",
    "local_magnitude": "ML",
    "duration_magnitude": "MD",
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

# get_station kwargs which are UTCDateTime objects
GET_STATIONS_UTC_KWARGS = (
    "starttime",
    "endtime",
    "startbefore",
    "startafter",
    "endbefore",
    "endafter",
)

# Numpy int types
NUMPY_INT_TYPES = {
    np.int,
    np.int32,
    np.int64,
    np.uint,
    np.dtype("int32"),
    np.dtype("int64"),
}

# Numpy float types
NUMPY_FLOAT_TYPES = {
    np.float32,
    np.float64,
    np.float16,
    # np.float128,  # windows raises error with this float type
    np.dtype("float32"),
    np.dtype("float64"),
    # np.dtype("float128"),
}

# str for types of data

DATA_TYPES = ("waveform", "station", "event")

# dtypes for the stats columns in waveframe

WAVEFRAME_STATS_DTYPES = {
    "network": str,
    "station": str,
    "location": str,
    "channel": str,
    "starttime": "ops_datetime",
    "endtime": "ops_datetime",
    "delta": "ops_timedelta",
    "sampling_rate": float,
}

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

# Columns of dataframes that are always date times
TIME_COLUMNS = (
    "time",
    "creation_time",
    "updated",  # I'm hoping this is actually the case
    "updatedafter",
    "end_date",
    "start_date",
    "event_time",
    "origin_time",
    "reference",
    "starttime",
    "endtime",
)

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
latitude: float, optional
    Specify the latitude to be used for a radius search.
longitude: float, optional
    Specify the longitude to the used for a radius search.
minradius: float, optional
    Limit to events within the specified minimum number of degrees from
    the geographic point defined by the latitude and longitude parameters.
maxradius: float, optional
    Limit to events within the specified maximum number of degrees from
    the geographic point defined by the latitude and longitude parameters.
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
    Limit to events updated after the specified time.
degrees: int, default True
    If False, the parameters maxradius and minradius are specified in m
    rather than degrees. Note: this parameter may not be supported by
    non-obsplus event clients.
"""

# the description for the parameter 'bar' in the bank `update_index` methods.
bar_parameter_description = """
bar
    This parameter controls if a progress bar will be used for this
    function call. Its behavior is dependent on the `bar` parameter:
        False - Don't use a progress bar
        None - Use the default progress bar
        ProgressBar - a custom implementation of progress bar is used.

    If a custom progress bar is to be used, it must have an `update`
    and `finish` method.
"""

paths_description = """
sub_paths
    A str, or iterable of str, specifying subdirectories (relative
    to bank path) to allow updating only files in specific directories
    of the bank. This is useful for large banks which have files added
    to them in predictable locations. However, if other files are added
    outside of these locations they may not get indexed as the banks timestamp
    indicating the last time of indexing will still get updated.
"""

# description for waveframe's starttime/endtime parameters
starttime_endtime_params = """
starttime
    Either a single value, or an array of values, indicating the
    start time of the new trace. Can also be ``np.timedelta64``
    object to reference a start time relative to current starttimes.
endtime
    Either a single value, or an array of values, indicating the
    end time of the new trace. Can also be ``np.timedelta64``
    object to reference an end time relative to current endtime.
"""
