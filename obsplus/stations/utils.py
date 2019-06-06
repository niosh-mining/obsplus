"""
Utilities for working with inventories.
"""
import inspect
from functools import singledispatch
from pathlib import Path

import obspy
from obspy.core.inventory import Channel, Station, Network

import obsplus
from obsplus.constants import station_clientable_type
from obsplus.interfaces import StationClient

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
        else:
            for ind, df_sub in df.groupby(columns):
                yield ind, df_sub

    def _get_kwargs(series, key_mapping):
        """ create the kwargs from a series and key mapping. """

        return {k: series.get(v) for k, v in key_mapping.items() if v in series}

    # first get key_mappings
    net_map = _make_key_mappings(Network)
    sta_map = _make_key_mappings(Station)
    cha_map = _make_key_mappings(Channel)
    # next define columns groupbys should be performed on
    net_columns = ["network", "start_date", "end_date"]
    sta_columns = ["station"]
    cha_columns = ["channel", "location"]
    # Ensure input is a dataframe
    df = obsplus.stations_to_df(df)
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

    return obspy.Inventory(networks=networks)


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
