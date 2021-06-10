"""
Module for management of seismic events.
"""
from typing_extensions import Annotated

import numpy as np
import obsplus.events.schema as eschema
import pandas as pd

from obsplus.utils.geodetics import map_longitudes
from obsplus.structures.mill import Mill
from obsplus.structures.dataframer import DataFramer
from obsplus.utils.time import to_datetime64


class EventMill(Mill):
    """
    A class for managing Seismic events.
    """

    _model = eschema.Catalog
    _id_name = "ResourceIdentifier"

    def _post_df_dict(self, df_dicts):
        """Run a few common sense checks on dfs."""
        # set preferred ids
        self._fill_preferred(df_dicts)
        return df_dicts

    def _fill_preferred(self, df_dicts=None, index=-1):
        """
        Fill in all the preferred_{whatever} ids on the event level.

        Parameters
        ----------
        index
            If the preferred is not set, use index to get it from a list.

        Returns
        -------

        """
        df_dicts = df_dicts if df_dicts is not None else self._df_dicts
        event_df = df_dicts["Event"]
        for name in {"origin", "magnitude", "focal_mechanism"}:
            preferred_id_name = f"preferred_{name}_id"
            object_column_name = f"{name}s"
            id_column = event_df[preferred_id_name]
            object_column = event_df[object_column_name]
            missing_preferred = ~id_column.astype(bool)
            has_objects = object_column.astype(bool)
            to_fill = missing_preferred & has_objects

            new_ids = object_column[to_fill].apply(lambda x: x[index])
            event_df.loc[to_fill, preferred_id_name] = new_ids
        return df_dicts


@EventMill.register_data_framer("events")
class EventFramer(DataFramer):
    """
    Class for extracting dataframes from event info from EventMill.
    """

    # Event level attrs
    _model = eschema.Event
    event_description: str = _model.event_descriptions[0].text
    event_id = _model.resource_id
    # origin attrs
    _origin: eschema.Origin = _model.preferred_origin_id.lookup()
    event_time: Annotated[np.datetime64, to_datetime64] = _origin.time
    event_latitude: float = _origin.latitude
    event_longitude: Annotated[float, map_longitudes] = _origin.longitude
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
    #
    # @model_operator
    # def _preferred_origin(self, object, mill):
    #     return self._get_preferred("origin", object, mill)
    #
    # @model_operator
    # def _preferred_magnitude(self, object, mill):
    #     return self._get_preferred("magnitude", object, mill)
    #
    # @model_operator
    # def _preferred_origin(self, object, mill):
    #     return self._get_preferred("origin", object, mill)
    #
    # def _get_preferred(self, what, object, mill):
    #     """Fetches one preferred thing or another."""
    #     breakpoint()
