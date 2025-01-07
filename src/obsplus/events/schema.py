"""
Pydantic schema for ObsPlus event model.

ObsPlus Event Model is a superset of, and compatible with, ObsPy's Event
model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4

import obspy.core.event as ev
from obspy import UTCDateTime
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainValidator,
    model_validator,
)

from obsplus.constants import NSLC

# ----- Type Literals (enum like)

data_used_wave_type = Literal[
    "P waves", "body waves", "surface waves", "mantle waves", "combined", "unknown"
]

AmplitudeCategory = Literal["point", "mean", "duration", "period", "integral", "other"]

AmplitudeUnit = Literal["m", "s", "m/s", "m/(s*s)", "m*s", "dimensionless", "other"]

DataUsedWaveType = Literal[
    "P waves", "body waves", "surface waves", "mantle waves", "combined", "unknown"
]

EvaluationMode = Literal["manual", "automatic"]

EvaluationStatus = Literal["preliminary", "confirmed", "reviewed", "final", "rejected"]

EventDescriptionType = Literal[
    "felt report",
    "Flinn-Engdahl region",
    "local time",
    "tectonic summary",
    "nearest cities",
    "earthquake name",
    "region name",
]

EventType = Literal[
    "not existing",
    "not reported",
    "earthquake",
    "anthropogenic event",
    "collapse",
    "cavity collapse",
    "mine collapse",
    "building collapse",
    "explosion",
    "accidental explosion",
    "chemical explosion",
    "controlled explosion",
    "experimental explosion",
    "industrial explosion",
    "mining explosion",
    "quarry blast",
    "road cut",
    "blasting levee",
    "nuclear explosion",
    "induced or triggered event",
    "rock burst",
    "reservoir loading",
    "fluid injection",
    "fluid extraction",
    "crash",
    "plane crash",
    "train crash",
    "boat crash",
    "other event",
    "atmospheric event",
    "sonic boom",
    "sonic blast",
    "acoustic noise",
    "thunder",
    "avalanche",
    "snow avalanche",
    "debris avalanche",
    "hydroacoustic event",
    "ice quake",
    "slide",
    "landslide",
    "rockslide",
    "meteorite",
    "volcanic eruption",
]

EventTypeCertainty = Literal["known", "suspected"]

MTInversionType = Literal["general", "zero trace", "double couple"]

MomentTensorCategory = Literal["teleseismic", "regional"]

OriginDepthType = Literal[
    "from location",
    "from moment tensor inversion",
    "from modeling of broad-band P waveforms",
    "constrained by depth phases",
    "constrained by direct phases",
    "constrained by depth and direct phases",
    "operator assigned",
    "other",
]

OriginType = Literal[
    "hypocenter",
    "centroid",
    "amplitude",
    "macroseismic",
    "rupture start",
    "rupture end",
]

OriginUncertaintyDescription = Literal[
    "horizontal uncertainty", "uncertainty ellipse", "confidence ellipsoid"
]

PickOnset = Literal["emergent", "impulsive", "questionable"]

PickPolarity = Literal["positive", "negative", "undecidable"]

SourceTimeFunctionType = Literal["box car", "triangle", "trapezoid", "unknown"]


def _to_datetime(dt: datetime | UTCDateTime) -> datetime:
    """Convert object to datatime."""
    return UTCDateTime(dt).datetime


UTCDateTimeFormat = Annotated[UTCDateTime, PlainValidator(_to_datetime)]

# ----- Type Models


class _ObsPyModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        from_attributes=True,
        extra="ignore",
    )

    # extra: Optional[AttribDictType] = None

    @staticmethod
    def _convert_to_obspy(value):
        """Convert an object to obspy or return value."""
        if hasattr(value, "to_obspy"):
            return value.to_obspy()
        return value

    def to_obspy(self):
        """Convert to obspy objects."""
        name = self.__class__.__name__
        cls = getattr(ev, name)
        # Note: converting to a dict is deprecated, but we don't want
        # to model dump because that is recursive, so we use this
        # ugly hack to just get all attributes
        out = {}
        for i in self.model_fields:
            val = getattr(self, i)
            if isinstance(val, list | tuple):
                out[i] = [self._convert_to_obspy(x) for x in val]
            else:
                out[i] = self._convert_to_obspy(val)
        return cls(**out)


class ResourceIdentifier(_ObsPyModel):
    """Resource ID"""

    id: str = Field(default_factory=lambda: str(uuid4()))


class _ModelWithResourceID(_ObsPyModel):
    """A model which has a resource ID"""

    resource_id: ResourceIdentifier = Field(
        default_factory=lambda: ResourceIdentifier()
    )


class QuantityError(_ObsPyModel):
    """Quantity Error"""

    uncertainty: float | None = None
    lower_uncertainty: float | None = None
    upper_uncertainty: float | None = None
    confidence_level: float | None = None


class CreationInfo(_ObsPyModel):
    """Creation info"""

    agency_id: str | None = None
    agency_uri: ResourceIdentifier | None = None
    author: str | None = None
    author_uri: ResourceIdentifier | None = None
    creation_time: UTCDateTimeFormat | None = None
    version: str | None = None


class TimeWindow(_ObsPyModel):
    """Time Window"""

    begin: float | None = None
    end: float | None = None
    reference: UTCDateTimeFormat | None = None


class CompositeTime(_ObsPyModel):
    """Composite Time"""

    year: int | None = None
    year_errors: QuantityError | None = None
    month: int | None = None
    month_errors: QuantityError | None = None
    day: int | None = None
    day_errors: QuantityError | None = None
    hour: int | None = None
    hour_errors: QuantityError | None = None
    minute: int | None = None
    minute_errors: QuantityError | None = None
    second: float | None = None
    second_errors: QuantityError | None = None


class Comment(_ModelWithResourceID):
    """Comment"""

    text: str | None = None
    creation_info: CreationInfo | None = None


class WaveformStreamID(_ObsPyModel):
    """Waveform stream ID"""

    network_code: str | None = None
    station_code: str | None = None
    location_code: str | None = None
    channel_code: str | None = None
    resource_uri: ResourceIdentifier | None = None
    seed_string: str | None = None

    @model_validator(mode="before")
    @classmethod
    def parse_seed_id(cls, values):
        """Parse seed IDs if needed."""
        seed_str = values.get("seed_string", None)
        # need to add seed_str
        if not seed_str:
            seed = ".".join([values[f"{x}_code"] for x in NSLC])
            values["seed_string"] = seed
            return values
        # need to get other codes from seed_str
        split = seed_str.split(".")
        for code, name in zip(split, NSLC):
            values[f"{name}_code"] = code
        return values


class ConfidenceEllipsoid(_ObsPyModel):
    """Confidence Ellipsoid"""

    semi_major_axis_length: float | None = None
    semi_minor_axis_length: float | None = None
    semi_intermediate_axis_length: float | None = None
    major_axis_plunge: float | None = None
    major_axis_azimuth: float | None = None
    major_axis_rotation: float | None = None


class DataUsed(_ObsPyModel):
    """Data Used"""

    wave_type: DataUsedWaveType | None = None
    station_count: int | None = None
    component_count: int | None = None
    shortest_period: float | None = None
    longest_period: float | None = None


# --- Magnitude classes


class StationMagnitude(_ModelWithResourceID):
    """Station Magnitude."""

    origin_id: ResourceIdentifier | None = None
    mag: float | None = None
    mag_errors: QuantityError | None = None
    station_magnitude_type: str | None = None
    amplitude_id: ResourceIdentifier | None = None
    method_id: ResourceIdentifier | None = None
    waveform_id: WaveformStreamID | None = None
    creation_info: CreationInfo | None = None
    comments: list[Comment] = []


class StationMagnitudeContribution(_ObsPyModel):
    """Station Magnitude Contribution"""

    station_magnitude_id: ResourceIdentifier | None = None
    residual: float | None = None
    weight: float | None = None


class Amplitude(_ModelWithResourceID):
    """Amplitude"""

    generic_amplitude: float | None = None
    generic_amplitude_errors: QuantityError | None = None
    type: str | None = None
    category: AmplitudeCategory | None = None
    unit: AmplitudeUnit | None = None
    method_id: ResourceIdentifier | None = None
    period: float | None = None
    period_errors: QuantityError | None = None
    snr: float | None = None
    time_window: TimeWindow | None = None
    pick_id: ResourceIdentifier | None = None
    waveform_id: WaveformStreamID | None = None
    filter_id: ResourceIdentifier | None = None
    scaling_time: UTCDateTimeFormat | None = None
    scaling_time_errors: QuantityError | None = None
    magnitude_hint: str | None = None
    evaluation_mode: EvaluationMode | None = None
    evaluation_status: EvaluationStatus | None = None
    creation_info: CreationInfo | None = None
    comments: list[Comment] = []


# --- Origin classes


class OriginUncertainty(_ObsPyModel):
    """Origin Uncertainty"""

    horizontal_uncertainty: float | None = None
    min_horizontal_uncertainty: float | None = None
    max_horizontal_uncertainty: float | None = None
    azimuth_max_horizontal_uncertainty: float | None = None
    confidence_ellipsoid: ConfidenceEllipsoid | None = None
    preferred_description: OriginUncertaintyDescription | None = None
    confidence_level: float | None = None


class OriginQuality(_ObsPyModel):
    """Origin Quality"""

    associated_phase_count: int | None = None
    used_phase_count: int | None = None
    associated_station_count: int | None = None
    used_station_count: int | None = None
    depth_phase_count: int | None = None
    standard_error: float | None = None
    azimuthal_gap: float | None = None
    secondary_azimuthal_gap: float | None = None
    ground_truth_level: str | None = None
    minimum_distance: float | None = None
    maximum_distance: float | None = None
    median_distance: float | None = None


class Pick(_ModelWithResourceID):
    """Pick"""

    time: UTCDateTimeFormat | None = None
    time_errors: QuantityError | None = None
    waveform_id: WaveformStreamID | None = None
    filter_id: ResourceIdentifier | None = None
    method_id: ResourceIdentifier | None = None
    horizontal_slowness: float | None = None
    horizontal_slowness_errors: QuantityError | None = None
    backazimuth: float | None = None
    backazimuth_errors: QuantityError | None = None
    slowness_method_id: ResourceIdentifier | None = None
    onset: PickOnset | None = None
    phase_hint: str | None = None
    polarity: PickPolarity | None = None
    evaluation_mode: EvaluationMode | None = None
    evaluation_status: EvaluationStatus | None = None
    creation_info: CreationInfo | None = None

    comments: list[Comment] = []


class Arrival(_ModelWithResourceID):
    """Arrival"""

    pick_id: ResourceIdentifier | None = None
    phase: str | None = None
    time_correction: float | None = None
    azimuth: float | None = None
    distance: float | None = None
    takeoff_angle: float | None = None
    takeoff_angle_errors: QuantityError | None = None
    time_residual: float | None = None
    horizontal_slowness_residual: float | None = None
    backazimuth_residual: float | None = None
    time_weight: float | None = None
    horizontal_slowness_weight: float | None = None
    backazimuth_weight: float | None = None
    earth_model_id: ResourceIdentifier | None = None
    creation_info: CreationInfo | None = None

    comments: list[Comment] = []


class Origin(_ModelWithResourceID):
    """Origin"""

    time: UTCDateTimeFormat
    time_errors: QuantityError | None = None
    longitude: float | None = None
    longitude_errors: QuantityError | None = None
    latitude: float | None = None
    latitude_errors: QuantityError | None = None
    depth: float | None = None
    depth_errors: QuantityError | None = None
    depth_type: OriginDepthType | None = None
    time_fixed: bool | None = None
    epicenter_fixed: bool | None = None
    reference_system_id: ResourceIdentifier | None = None
    method_id: ResourceIdentifier | None = None
    earth_model_id: ResourceIdentifier | None = None
    quality: OriginQuality | None = None
    origin_type: OriginType | None = None
    origin_uncertainty: OriginUncertainty | None = None
    region: str | None = None
    evaluation_mode: EvaluationMode | None = None
    evaluation_status: EvaluationStatus | None = None
    creation_info: CreationInfo | None = None

    comments: list[Comment] = []
    arrivals: list[Arrival] = []
    composite_times: list[CompositeTime] = []


class Magnitude(_ModelWithResourceID):
    """Magnitude"""

    mag: float | None = None
    mag_errors: QuantityError | None = None
    magnitude_type: str | None = None
    origin_id: ResourceIdentifier | None = None
    method_id: ResourceIdentifier | None = None
    station_count: int | None = None
    azimuthal_gap: float | None = None
    evaluation_mode: EvaluationMode | None = None
    evaluation_status: EvaluationStatus | None = None
    creation_info: CreationInfo | None = None
    comments: list[Comment] = []
    station_magnitude_contributions: list[StationMagnitudeContribution] = []


# --- Source objects


class Axis(_ObsPyModel):
    """Axis"""

    azimuth: float | None = None
    plunge: float | None = None
    length: float | None = None


class NodalPlane(_ObsPyModel):
    """Nodal Plane"""

    strike: float | None = None
    dip: float | None = None
    rake: float | None = None


class NodalPlanes(_ObsPyModel):
    """Nodal Planes"""

    nodal_plane_1: NodalPlane | None = None
    nodal_plane_2: NodalPlane | None = None
    preferred_plane: int | None = None


class PrincipalAxes(_ObsPyModel):
    """Principal Axes"""

    t_axis: Axis | None = None
    p_axis: Axis | None = None
    n_axis: Axis | None = None


class Tensor(_ObsPyModel):
    """Tensor"""

    m_rr: float | None = None
    m_rr_errors: QuantityError | None = None
    m_tt: float | None = None
    m_tt_errors: QuantityError | None = None
    m_pp: float | None = None
    m_pp_errors: QuantityError | None = None
    m_rt: float | None = None
    m_rt_errors: QuantityError | None = None
    m_rp: float | None = None
    m_rp_errors: QuantityError | None = None
    m_tp: float | None = None
    m_tp_errors: QuantityError | None = None


class SourceTimeFunction(_ObsPyModel):
    """Source Time Function"""

    type: SourceTimeFunctionType | None = None
    duration: float | None = None
    rise_time: float | None = None
    decay_time: float | None = None


class MomentTensor(_ModelWithResourceID):
    """Moment Tensor"""

    derived_origin_id: ResourceIdentifier | None = None
    moment_magnitude_id: ResourceIdentifier | None = None
    scalar_moment: float | None = None
    scalar_moment_errors: QuantityError | None = None
    tensor: Tensor | None = None
    variance: float | None = None
    variance_reduction: float | None = None
    double_couple: float | None = None
    clvd: float | None = None
    iso: float | None = None
    greens_function_id: float | None = None
    filter_id: ResourceIdentifier | None = None
    source_time_function: SourceTimeFunction | None = None
    data_used: list[DataUsed] | None = None
    method_id: ResourceIdentifier | None = None
    category: MomentTensorCategory | None = None
    inversion_type: MTInversionType | None = None
    creation_info: CreationInfo | None = None


class FocalMechanism(_ModelWithResourceID):
    """Focal Mechanism"""

    triggering_origin_id: ResourceIdentifier | None = None
    nodal_planes: NodalPlanes | None = None
    principal_axes: PrincipalAxes | None = None
    azimuthal_gap: float | None = None
    station_polarity_count: int | None = None
    misfit: float | None = None
    station_distribution_ratio: float | None = None
    method_id: ResourceIdentifier | None = None
    evaluation_mode: EvaluationMode | None = None
    evaluation_status: EvaluationStatus | None = None
    moment_tensor: MomentTensor | None = None
    creation_info: CreationInfo | None = None

    waveform_id: list[WaveformStreamID] = []
    comments: list[Comment] = []


# --- Event definitions


class EventDescription(_ObsPyModel):
    """Event Description"""

    text: str | None = None
    type: EventDescriptionType | None = None


class Event(_ModelWithResourceID):
    """Event"""

    event_type: EventType | None = None
    event_type_certainty: EventTypeCertainty | None = None
    creation_info: CreationInfo | None = None
    preferred_origin_id: ResourceIdentifier | None = None
    preferred_magnitude_id: ResourceIdentifier | None = None
    preferred_focal_mechanism_id: ResourceIdentifier | None = None
    event_descriptions: list[EventDescription] = []
    comments: list[Comment] = []
    picks: list[Pick] = []
    amplitudes: list[Amplitude] = []
    focal_mechanisms: list[FocalMechanism] = []
    origins: list[Origin] = []
    magnitudes: list[Magnitude] = []
    station_magnitudes: list[StationMagnitude] = []

    def to_obspy(self):
        """Convert the catalog to obspy form"""
        out = super().to_obspy()
        out.scope_resource_ids()
        return out


class Catalog(_ModelWithResourceID):
    """A collection of events."""

    events: list[Event] = []
    description: str | None = None
    comments: list[Comment] | None = None
    creation_info: CreationInfo | None = None
