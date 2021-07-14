"""
Module for management of seismic events.
"""
import obspy
from typing_extensions import Annotated

import numpy as np
import obsplus.events.schema as eschema
import pandas as pd

from obsplus.constants import get_events_parameters
from obsplus.utils.geodetics import map_longitudes
from obsplus.structures.mill import Mill, MillType, _inplace_or_copy
from obsplus.structures.dataframer import DataFramer
from obsplus.utils.time import to_datetime64
from obsplus.utils.pd import get_index_group
from obsplus.events.get_events import (
    _dict_times_to_npdatetimes,
    _get_ids,
    _validate_get_event_kwargs,
)
from obsplus.utils.docs import compose_docstring

TIME_TYPE = Annotated[np.datetime64, to_datetime64]


class EventMill(Mill):
    """
    A class for managing Seismic events.
    """

    _model = eschema.Catalog
    _id_name = "ResourceIdentifier"
    _index_group = ("parent_id", "attr")
    _scope_model_name = "Event"

    @_inplace_or_copy
    def fill_preferred(self: MillType, index=-1, inplace=False) -> MillType:
        """
        Fill in all the preferred_{whatever} ids on the event level.

        Parameters
        ----------
        index
            If the preferred is not set, use index to get it from a list.
        """
        struct = self.get_df("__structure__")
        event_df = self.get_df("Event")
        for name in {"origin", "magnitude", "focal_mechanism"}:
            preferred_id_name = f"preferred_{name}_id"
            id_column = event_df[preferred_id_name]
            # determine which columns are missing, if none just return
            missing_preferred = ~id_column.astype(bool)
            if not missing_preferred.any():
                continue
            # get model type from schema
            event_schema = self._schema_df["schema_Event"]["referenced_model"]
            model_name = event_schema[preferred_id_name]
            # get potential preferred objects
            sub_struct = struct[
                (struct["scope_id"].isin(missing_preferred.index))
                & (struct["model"] == model_name)
            ]
            # get df of child type
            column_group = ("scope_id", "attr", "model")
            sub_inds = get_index_group(
                sub_struct,
                index=index,
                column_group=column_group,
            )
            new_id_series = (
                sub_struct.loc[sub_inds]
                .reset_index()
                .set_index("scope_id")["resource_id"]
            )
            event_df.loc[:, preferred_id_name] = new_id_series
        return self

    @compose_docstring(params=get_events_parameters)
    def get_events(self, **kwargs) -> obspy.Catalog:
        """
        Extract an ObsPy Catalog from EventMill.

        Parameters
        ----------
        {params}
        """
        df = self.get_summary_df()
        _validate_get_event_kwargs(kwargs, extra=set(df.columns))
        kwargs = _dict_times_to_npdatetimes(kwargs)
        event_ids = _get_ids(df, kwargs)
        return self.to_model(scope_ids=event_ids).to_obspy()

    def _get_summary(self, df_dicts):
        """Add a summary table to the dataframe dict for easy querying."""
        return self.get_df("event_core")


@EventMill.register_data_framer("event_core")
class CoreEventFramer(DataFramer):
    """
    Framer to get the minimum required info from events.

    Only extracts enough information to support queries using `get_event.
    """

    # Event level attrs
    _model = eschema.Event
    _origin: eschema.Origin = _model.preferred_origin_id
    _pref_mag = _model.preferred_magnitude_id
    _quality: eschema.OriginQuality = _origin.quality
    _creation_info = _model.creation_info

    time: TIME_TYPE = _origin.time
    latitude: float = _origin.latitude
    longitude: Annotated[float, map_longitudes] = _origin.longitude
    depth: float = _origin.depth
    magnitude: float = _pref_mag.mag
    magnitude_type: str = _pref_mag.magnitude_type
    event_description: str = _model.event_descriptions[0].text
    event_id: str = _model.resource_id
    updated: Annotated[np.datetime64, to_datetime64] = _creation_info.creation_time
    standard_error: float = _quality.standard_error


@EventMill.register_data_framer("events")
class EventFramer(CoreEventFramer):
    """
    Framer to get the rest of the event information that might be useful.
    """

    _model: eschema.Event = CoreEventFramer._model
    _origin: eschema.Origin = _model.preferred_origin_id
    _pref_mag: eschema.Magnitude = _model.preferred_magnitude_id
    _quality: eschema.OriginQuality = _origin.quality
    _uncertainty: eschema.OriginUncertainty = _origin.origin_uncertainty
    _mags = _model.magnitudes
    _creation_info = _model.creation_info
    # origin quality stuff
    used_phase_count: pd.Int64Dtype() = _quality.associated_phase_count
    used_station_count: pd.Int64Dtype() = _quality.used_station_count
    horizontal_uncertainty: float = _uncertainty.horizontal_uncertainty
    azimuthal_gap: float = _quality.azimuthal_gap
    # magnitude stuff
    local_magnitude: float = _mags.match(magnitude_type="ML").mag.last()
    moment_magnitude: float = _mags.match(magnitude_type="Mw").mag.last()
    duration_magnitude: float = _mags.match(magnitude_type="Md").mag.last()
    updated: TIME_TYPE = _creation_info.creation_time
    author: str = _creation_info.author
    agency_id: str = _creation_info.agency_id
    version: str = _creation_info.version


@EventMill.register_data_framer("picks")
class PickFramer(DataFramer):
    """Dataframer for extracting picks from mill."""

    _model = eschema.Pick
    resource_id = _model.resource_id
    time = _model.time
    polarity = _model.polarity
    phase_hint = _model.phase_hint
    event_id = _model.parent().resource_id
