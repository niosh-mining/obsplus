"""
Utilities for calculating spatial relationships.
"""

from typing import Union

import numpy as np
import pandas as pd
from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics.base import WGS84_A, WGS84_F
from obspy.taup.taup_geo import calc_dist

import obsplus
from obsplus.constants import (
    event_type,
    inventory_type,
    DISTANCE_COLUMN_DTYPES,
    DISTANCE_COLUMN_INPUT_DTYPES,
)
from obsplus.utils.docs import compose_docstring
from obsplus.exceptions import DataFrameContentError


class SpatialCalculator:
    """
    Class for calculating spatial relationships between two entities.

    The default values are appropriate for Earth.

    Parameters
    ----------
    radius
        The radius of the planetary body in question.
    flattening
        The flattening coefficient of the planetary body in question.

    Examples
    --------
    >>> import obsplus
    >>> import obspy
    >>> from obsplus.utils import SpatialCalculator
    >>> calc = SpatialCalculator()  # SpatialCalculator for Earth
    >>> # Get the distance and azimuth between two points
    >>> p1 = (45.55, -111.21, 1000)  # format is lat, lon, elevation (m asl)
    >>> p2 = (40.22, -115, 10)
    >>> df = calc(p1, p2)

    >>> # Get the distance between each event and each station
    >>> events = obspy.read_events()
    >>> station = obspy.read_inventory()
    >>> df = calc(events, station)
    >>> # index 1 is the event_id and index 2 is the seed id
    """

    expected_exceptions = (TypeError, ValueError, AttributeError)

    def __init__(self, radius: float = WGS84_A, flattening: float = WGS84_F):
        self.radius = radius
        self.flattening = flattening

    # --- methods for getting input dataframe from various objects

    def _get_dataframe(self, obj) -> pd.DataFrame:
        """
        Return a dataframe with latitude, longitude, elevation, and id.
        """
        cols = list(DISTANCE_COLUMN_INPUT_DTYPES)
        # if a dataframe is used
        if isinstance(obj, pd.DataFrame) and set(cols).issubset(obj.columns):
            return self._validate_dataframe(obj)
        try:  # first try events
            df = self._df_from_events(obj)
        except self.expected_exceptions:  # then stations
            try:
                df = self._df_from_stations(obj)
            except self.expected_exceptions:
                # and lastly any sequence.
                df = self._df_from_sequences(obj)
        return self._validate_dataframe(df)

    def _df_from_events(self, obj):
        """ Get the needed dataframe from some objects with event data. """
        df = obsplus.events_to_df(obj).set_index("event_id")
        df["elevation"] = -df["depth"]
        return df

    def _df_from_stations(self, obj):
        """ Get the needed dataframe from some object with station data. """
        df = obsplus.stations_to_df(obj).set_index("seed_id")
        return df

    def _df_from_sequences(self, obj):
        """ Get the dataframe from generic sequences. """
        ar = np.atleast_2d(obj)
        if ar.shape[1] == 3:  # need to add index columns
            id = np.arange(len(ar))
        elif ar.shape[1] == 4:
            ar, id = ar[:, :3].astype(float), ar[:, 3]
        else:
            msg = "A sequence must have either 3 or 4 elements."
            raise ValueError(msg)
        cols = list(DISTANCE_COLUMN_INPUT_DTYPES)
        df = pd.DataFrame(ar, index=id, columns=cols)
        return df

    def _de_duplicate_df_index(self, df) -> pd.DataFrame:
        """
        Remove any duplicated indices, raise an exception if any of the
        duplicated indices have differing coordinates.
        """
        # determine if there are any duplicates with different coords
        duplicate_indices = df.index.duplicated()
        duplicated_data = df.duplicated(list(DISTANCE_COLUMN_INPUT_DTYPES))
        # if so raise an exception as this will not produced desired result.
        if (duplicate_indices & (~duplicated_data)).any():
            dup_ids = set(df.index[(duplicate_indices & (~duplicated_data))])
            msg = (
                f"There are multiple coordinates for ids: {dup_ids} "
                "cannot reliable calculate distance relationships."
            )
            raise ValueError(msg)
        return df[~duplicate_indices]

    def _validate_dataframe(self, df) -> pd.DataFrame:
        """ Ensure all the parameters of the dataframe are reasonable. """
        # first cull out columns that aren't needed and de-dup index
        out = (
            df[list(DISTANCE_COLUMN_INPUT_DTYPES)]
            .astype(DISTANCE_COLUMN_INPUT_DTYPES)
            .pipe(self._de_duplicate_df_index)
        )
        # sanity checks on lat/lon
        lat_valid = abs(df["latitude"]) <= 90.0
        lons_valid = abs(df["longitude"]) <= 180.0
        if not (lat_valid.all() & lons_valid.all()):
            msg = f"invalid lat/lon values found in {df}"
            raise DataFrameContentError(msg)
        return out

    # --- methods for calculating spatial relationships

    def _get_spatial_relations(self, array):
        """ Calculate distances and azimuths. """
        # TODO this may be vectorizable, look into it
        # get planet radius and flattening
        a, f = self.radius, self.flattening
        rad_km = a / 1000.0
        # first calculate the great circle distance in m, forward az and back az
        m_az1_az2 = [gps2dist_azimuth(x[0], x[1], x[3], x[4], a=a, f=f) for x in array]
        # then calculate distance in degrees
        dist_degrees = [calc_dist(x[0], x[1], x[3], x[4], rad_km, f) for x in array]
        # vertical distances
        vert = array[:, 2] - array[:, 5]
        # stack and return
        out1 = np.array(m_az1_az2)
        out2 = np.array(dist_degrees).reshape(-1, 1)
        return np.hstack([out1, out2, vert.reshape(-1, 1)])

    @compose_docstring(
        out_columns=str(tuple(DISTANCE_COLUMN_DTYPES)),
        in_columns=str(tuple(DISTANCE_COLUMN_INPUT_DTYPES)),
    )
    def __call__(
        self,
        entity_1: Union[event_type, inventory_type, pd.DataFrame, tuple],
        entity_2: Union[event_type, inventory_type, pd.DataFrame, tuple],
    ) -> pd.DataFrame:
        """
        Calculate spatial relationship(s) between two entities.

        Parameters
        ----------
        entity_1
            A variety of obspy/obsplus types which have linked geographic info.
        entity_2
            A variety of obspy/obsplus types which have linked geographic info.

        Notes
        -----
        If a dataframe is used for input it must have the following columns:
        {in_columns}

        Returns
        -------
        A dataframe with rows for each combination of entities with columns:
        {out_columns}
        """
        # first convert each entity to dataframe with lat, lon, elevation as
        # columns and a meaningful id as the index.
        obj1 = self._get_dataframe(entity_1)
        obj2 = self._get_dataframe(entity_2)
        # first get cartesian product from both indices and make aligned dfs
        mesh = np.meshgrid(obj1.index.values, obj2.index.values)
        ind1, ind2 = mesh[0].flatten(), mesh[1].flatten()
        df1, df2 = obj1.loc[ind1], obj2.loc[ind2]
        assert len(df1) == len(df2)
        # create array of lat1, lon1, ele1, lat2, lon2, ele2
        array = np.hstack((df2.values, df1.values))
        # get spatial relationships, create index and return df
        out = self._get_spatial_relations(array)
        index = pd.MultiIndex.from_arrays([ind1, ind2], names=["id1", "id2"])
        return pd.DataFrame(out, columns=list(DISTANCE_COLUMN_DTYPES), index=index)
