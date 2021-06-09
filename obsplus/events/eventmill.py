"""
Module for management of seismic events.
"""
from typing_extensions import Annotated

import numpy as np
import obsplus.events.schema as eschema
from obsplus.structures.mill import Mill
from obsplus.structures.dataframer import DataFramer, model_operator
import pandas as pd


class EventMill(Mill):
    """
    A class for managing Seismic events.
    """

    _model = eschema.Catalog
    _id_name = "ResourceIdentifier"


@EventMill.register_data_framer("events")
class EventFramer(DataFramer):
    """
    Class for extracting dataframes from event info from EventMill.
    """

    # Event level attrs
    _model = eschema.Event
    event_description: str = _model.event_descriptions[0].text
    event_id = _model.resource_id.id
    # origin attrs
    _origin: eschema.Origin = _model._preferred_origin
    event_time: np.datetime64 = _origin.time
    event_latitude: float = _origin.latitude
    event_longitude: Annotated[float, "longitude"] = _origin.longitude
    event_depth: float = _origin.depth
    # magnitude attrs
    _pref_mag = _model._preferred_magnitude
    magnitude: float = _pref_mag.magnitude
    magnitude_type: str = _pref_mag.magnitude_type
    # origin quality attrs
    _origin_quality: eschema.OriginQuality = _origin.quality
    used_phase_count: pd.Int64Dtype() = _origin_quality.associated_phase_count
    used_station_count: pd.Int64Dtype() = _origin_quality.used_station_count
    standard_error: float = _origin_quality.standard_error
    azimuthal_gap: float = _origin_quality.azimuthal_gap

    @model_operator
    def _preferred_origin(self, object, mill):
        return self._get_preferred("origin", object, mill)

    @model_operator
    def _preferred_magnitude(self, object, mill):
        return self._get_preferred("magnitude", object, mill)

    @model_operator
    def _preferred_origin(self, object, mill):
        return self._get_preferred("origin", object, mill)

    def _get_preferred(self, what, object, mill):
        """Fetches one preferred thing or another."""
        breakpoint()
