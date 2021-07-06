"""
Module for management of seismic events.
"""
import copy
from typing_extensions import Annotated

import numpy as np
import obsplus.events.schema as eschema
import pandas as pd

from obsplus.utils.geodetics import map_longitudes
from obsplus.structures.mill import Mill, MillType
from obsplus.structures.dataframer import DataFramer
from obsplus.utils.time import to_datetime64
from obsplus.utils.pd import loc_by_name, get_index_group, expand_loc


TIME_TYPE = Annotated[np.datetime64, to_datetime64]


class EventMill(Mill):
    """
    A class for managing Seismic events.
    """

    _model = eschema.Catalog
    _id_name = "ResourceIdentifier"
    _index_group = ("parent_id", "attr")

    def fill_preferred(self: MillType, index=-1, inplace=False) -> MillType:
        """
        Fill in all the preferred_{whatever} ids on the event level.

        Parameters
        ----------
        index
            If the preferred is not set, use index to get it from a list.

        Returns
        -------

        """
        schema = self._model.get_obsplus_schema()
        df_dicts = self._table_dict if inplace else copy.deepcopy(self._table_dict)
        event_df = df_dicts["Event"]
        eids = event_df.index.get_level_values("resource_id")
        for name in {"origin", "magnitude", "focal_mechanism"}:
            preferred_id_name = f"preferred_{name}_id"
            object_column_name = f"{name}s"
            id_column = event_df[preferred_id_name]
            # determine which columns are missing, if non just return
            missing_preferred = ~id_column.astype(bool)
            if not missing_preferred.any():
                continue
            # get df of child type
            sub_class = schema["Event"]["attr_ref"][object_column_name]
            sub_df = df_dicts[sub_class]
            subs = loc_by_name(sub_df, scope_id=eids, attr=f"{name}s", index=-1)
            # get values from child table
            last_inds = get_index_group(subs, index, column_group=self._index_group)
            out = expand_loc(subs[last_inds], parent_id=eids.values)["resource_id"]
            # determine if values should be replaced
            should_fill = (missing_preferred & (~pd.isnull(out))).values
            event_df.loc[should_fill, preferred_id_name] = out.values[should_fill]
        return self._from_df_dict(df_dicts)


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

    time: TIME_TYPE = _origin.time
    latitude: float = _origin.latitude
    longitude: Annotated[float, map_longitudes] = _origin.longitude
    depth: float = _origin.depth
    magnitude: float = _pref_mag.mag
    magnitude_type: str = _pref_mag.magnitude_type
    event_description: str = _model.event_descriptions[0].text
    event_id: str = _model.resource_id
    updated: Annotated[np.datetime64, to_datetime64] = _model.creation_info.time
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
