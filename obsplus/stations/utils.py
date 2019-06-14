"""
Utilities for working with inventories.
"""
import inspect
from functools import singledispatch
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
from obspy.core.inventory import Channel, Station, Network

import obsplus
from obsplus.constants import station_clientable_type
from obsplus.interfaces import StationClient

LARGE_NUMBER = obspy.UTCDateTime("3000-01-01").timestamp

# create key mappings for mapping keys (used to create inventory objects
# from various columns in the stations dataframe).
mapping_keys = {
    Channel: {"code": "channel", "location_code": "location"},
    Station: {"code": "station"},
    Network: {"code": "network"},
}


def df_to_inventory(df) -> obspy.Inventory:
    """
    Create a simple inventory from a dataframe.

    The dataframe must have the same columns as the once produced by
    :func:`obsplus.stations_to_df`.
    """

    def _make_key_mappings(cls):
        """ Create a mapping from columns in df to kwargs for cls. """
        base_params = set(inspect.signature(cls).parameters)
        new_map = mapping_keys[cls]
        base_map = {x: x for x in base_params - set(new_map)}
        base_map.update(new_map)
        return base_map

    def _groupby_if_exists(df, columns):
        """ Groupby columns if they exist on dataframe, else return empty. """
        cols = list(obsplus.utils.iterate(columns))
        if not set(cols).issubset(df.columns):
            return

        # copy df and set missing start/end times to reasonable values
        # this is needed so they get included in a groupby
        df = df.copy()
        isnan = df.isna()
        if "start_date" in columns:
            df["start_date"] = df["start_date"].fillna(0)
        if "end_date" in columns:
            df["end_date"] = df["end_date"].fillna(LARGE_NUMBER)

        for ind, df_sub in df.groupby(cols):
            # replace NaN values
            if isnan.any().any():
                df_sub[isnan.loc[df_sub.index]] = np.nan
            yield ind, df_sub

    def _get_kwargs(series, key_mapping):
        """ create the kwargs from a series and key mapping. """

        out = {}
        for k, v in key_mapping.items():
            # skip if requested kwarg is not in the series
            if v not in series:
                continue
            value = series[v]
            out[k] = value if not pd.isnull(value) else None
        return out

    # first get key_mappings
    net_map = _make_key_mappings(Network)
    sta_map = _make_key_mappings(Station)
    cha_map = _make_key_mappings(Channel)
    # next define columns groupbys should be performed on
    net_columns = ["network"]
    sta_columns = ["station", "start_date", "end_date"]
    cha_columns = ["channel", "location", "start_date", "end_date"]
    # Ensure input is a dataframe
    df = obsplus.stations_to_df(df)
    # replace
    # Iterate networks and create stations
    networks = []
    for net_code, net_df in _groupby_if_exists(df, net_columns):
        stations = []
        for st_code, sta_df in _groupby_if_exists(net_df, sta_columns):
            channels = []
            for ch_code, ch_df in _groupby_if_exists(sta_df, cha_columns):
                kwargs = _get_kwargs(ch_df.iloc[0], cha_map)
                channels.append(Channel(**kwargs))
            kwargs = _get_kwargs(sta_df.iloc[0], sta_map)
            stations.append(Station(channels=channels, **kwargs))
        kwargs = _get_kwargs(net_df.iloc[0], net_map)
        networks.append(Network(stations=stations, **kwargs))

    return obspy.Inventory(networks=networks, source=f"ObsPlus_v{obsplus.__version__}")


@singledispatch
def get_station_client(stations: station_clientable_type) -> StationClient:
    """
    Extract a station client from various inputs.

    Parameters
    ----------
    stations
        Any of the following:
            * A path to an obspy-readable station file
            * A path to a directory of obspy-readable station files
            * An `obspy.Inventory` instance
            * An instance of :class:`~obsplus.EventBank`
            * Any other object that has a `get_stations` method

    Raises
    ------
    TypeError
        If a station client cannot be determined from the input.
    """
    if not isinstance(stations, StationClient):
        msg = f"a station client could not be extracted from {stations}"
        raise TypeError(msg)
    return stations


@get_station_client.register(Path)
@get_station_client.register(str)
def _read_inventory(path) -> StationClient:
    path = Path(path)
    inv = None
    if path.is_dir():
        for inv_path in path.rglob("*.xml"):
            if inv is None:
                inv = obspy.read_inventory(str(inv_path))
            else:
                inv += obspy.read_inventory(str(inv_path))
    else:
        inv = obspy.read_inventory(str(path))
    return get_station_client(inv)
