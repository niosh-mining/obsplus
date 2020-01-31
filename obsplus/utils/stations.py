"""
Utilities for working with inventories.
"""
import inspect
from functools import singledispatch, lru_cache
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
from obspy.core.inventory import Channel, Station, Network

import obsplus
import obsplus.utils.misc
from obsplus.constants import station_clientable_type, SMALLDT64, LARGEDT64, NSLC
from obsplus.interfaces import StationClient
from obsplus.utils.time import to_utc

LARGE_NUMBER = obspy.UTCDateTime("3000-01-01").timestamp

# create key mappings for mapping keys (used to create inventory objects
# from various columns in the stations dataframe).
mapping_keys = {
    Channel: {"code": "channel", "location_code": "location"},
    Station: {"code": "station"},
    Network: {"code": "network"},
}

type_mappings = {"start_date": to_utc, "end_date": to_utc}


def df_to_inventory(df) -> obspy.Inventory:
    """
    Create a station inventory from a dataframe.

    Parameters
    ----------
    df
        A dataframe which must have the same columns as the once produced by
        :func:`obsplus.stations_to_df`.

    Notes
    -----
    The dataframe can also contain columns named "sensor_keys" and
    "datalogger_keys" which will indicate the response information should
    be fetched suing obspy's ability to interact with the nominal response
    library. Each of these columns should either contain tuples or strings
    where the keys are separated by double underscores (__).
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
        cols = list(obsplus.utils.misc.iterate(columns))
        # copy df and set missing start/end times to reasonable values
        # this is needed so they get included in a groupby
        df = df.copy()
        isnan = df.isna()
        default_start = pd.Timestamp(SMALLDT64)
        default_end = pd.Timestamp(LARGEDT64)

        if "start_date" in columns:
            df["start_date"] = df["start_date"].fillna(default_start)
        if "end_date" in columns:
            df["end_date"] = df["end_date"].fillna(default_end)

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
            value = value if not pd.isnull(value) else None
            # if the type needs to be cast to something else
            if k in type_mappings and value is not None:
                value = type_mappings[k](value)
            out[k] = value

        return out

    @lru_cache()
    def get_nrl():
        """ Initiate a nominal response library object. """
        from obspy.clients.nrl import NRL

        return NRL()

    @lru_cache()
    def get_response(datalogger_keys, sensor_keys):
        nrl = get_nrl()
        kwargs = dict(datalogger_keys=datalogger_keys, sensor_keys=sensor_keys)
        return nrl.get_response(**kwargs)

    def _get_resp_key(key):
        """ Get response keys from various types. """
        if isinstance(key, str) or key is None:
            return tuple((key or "").split("__"))
        else:
            return tuple(key)

    def _maybe_add_response(series, channel_kwargs):
        """ Maybe add the response information if required columns exist. """
        # bail out of required columns do not exist
        if not {"sensor_keys", "datalogger_keys"}.issubset(set(series.index)):
            return
        # determine if both required columns are populated, else bail out
        sensor = series["sensor_keys"]
        datalogger = series["datalogger_keys"]
        if pd.isnull(sensor) or pd.isnull(datalogger):
            return
        sensor_keys = _get_resp_key(sensor)
        datalogger_keys = _get_resp_key(datalogger)
        # at this point all the required info for resp lookup should be there
        channel_kwargs["response"] = get_response(datalogger_keys, sensor_keys)

    # make sure all seed_id codes are str
    for col in set(NSLC) & set(df.columns):
        df[col] = df[col].astype(str).str.replace(".0", "")
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
    # Iterate networks and create stations
    networks = []
    for net_code, net_df in _groupby_if_exists(df, net_columns):
        stations = []
        for st_code, sta_df in _groupby_if_exists(net_df, sta_columns):
            if not st_code[0]:
                continue
            channels = []
            for ch_code, ch_df in _groupby_if_exists(sta_df, cha_columns):
                if not ch_code[0]:  # skip empty channel lines
                    continue
                chan_series = ch_df.iloc[0]
                kwargs = _get_kwargs(chan_series, cha_map)
                # try to add the inventory
                _maybe_add_response(chan_series, kwargs)
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
