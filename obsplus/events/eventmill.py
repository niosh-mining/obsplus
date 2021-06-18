"""
Module for management of seismic events.
"""
import copy
from typing import TypeVar
from typing_extensions import Annotated

import numpy as np
import obsplus.events.schema as eschema
import pandas as pd

from obsplus.utils.geodetics import map_longitudes
from obsplus.structures.mill import Mill, MillType
from obsplus.structures.dataframer import DataFramer
from obsplus.utils.time import to_datetime64
from obsplus.utils.pd import loc_by_name, get_index_group, expand_loc


class EventMill(Mill):
    """
    A class for managing Seismic events.
    """

    _model = eschema.Catalog
    _id_name = "ResourceIdentifier"
    _index_group = ('parent_id', 'attr')

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
        df_dicts = self._df_dicts if inplace else copy.deepcopy(self._df_dicts)
        event_df = df_dicts["Event"]
        eids = event_df.index.get_level_values('resource_id')
        for name in {"origin", "magnitude", "focal_mechanism"}:
            preferred_id_name = f"preferred_{name}_id"
            object_column_name = f"{name}s"
            id_column = event_df[preferred_id_name]
            # determine which columns are missing, if non just return
            missing_preferred = ~id_column.astype(bool)
            if not missing_preferred.any():
                continue
            # get df of child type
            sub_class = schema['Event']['attr_ref'][object_column_name]
            sub_df = df_dicts[sub_class]
            subs = loc_by_name(sub_df, scope_id=eids, attr=f"{name}s", index=-1)
            # get values from child table
            last_inds = get_index_group(subs, index, column_group=self._index_group)
            out = expand_loc(subs[last_inds], parent_id=eids.values)['resource_id']
            # determine if values should be replaced
            should_fill = (missing_preferred & (~pd.isnull(out))).values
            event_df.loc[should_fill, preferred_id_name] = out.values[should_fill]
        return self._from_df_dict(df_dicts)


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
