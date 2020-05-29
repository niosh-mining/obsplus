"""
Utilities for working with inventories.
"""
import inspect
import json
from functools import singledispatch, lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import obspy
import pandas as pd
from obspy.core.inventory import Channel, Station, Network

import obsplus
import obsplus.utils.misc
from obsplus.constants import (
    station_clientable_type,
    SMALLDT64,
    LARGEDT64,
    NSLC,
    GET_STATIONS_UTC_KWARGS,
)
from obsplus.exceptions import AmbiguousResponseError
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


def df_to_inventory(
    df: pd.DataFrame, client: Optional[StationClient] = None
) -> obspy.Inventory:
    """
    Create a station inventory from a dataframe.

    Parameters
    ----------
    df
        A dataframe which must have the same columns as the once produced by
        :func:`obsplus.stations_to_df`.
    client


    Notes
    -----
    There are two ways to include response information:

    1. Using the Nominal Response Library (NRL):
    (https://docs.obspy.org/master/packages/obspy.clients.nrl.html)
    If the dataframe has columns named "sensor_keys" and "datalogger_keys"
    these will indicate the response information should
    be fetched using ObsPy's NRL client. Each of these columns should either
    contain tuples or strings where the keys are separated by double
    underscores (__).
    For example, to specify sensor keys:
    ('Nanometrics', 'Trillium 120 Horizon')
    or
    'Nanometrics__Trillium 120 Horizon'
    are both valid.

    2. Using a station client:
    If the dataframe contains a column get_stations_kwargs it indicates that
    either a client was passed as the client argument or the fdsn IRIS client
    should be used to download station information. The contents of this column
    must be a dictionary of acceptable keyword arguments for the client's
    `get_stations` method. All time values must be provided as iso8601
    strings.
    For example,
    {'network': 'UU', 'station': 'TMU', 'location': '01', 'channel': 'HHZ',
     'starttime': '2017-01-02',}
    would use either a provided station client (or ObsPy's default FDSN client
    if non is provided) to download a response for the corresponding channel.
    - If more than one channel is returned from teh get_stations call an error
      will be raised and a more specific query will be required.
    """
    response_cols = {"get_station_kwargs", "sensor_keys", "datalogger_keys"}

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
    def get_station_client():
        """
        Instantiate an IRIS FDSN client or return the provided client.
        """
        from obspy.clients.fdsn import Client

        if client is not None:
            assert isinstance(client, StationClient)
            return client
        return Client()

    @lru_cache()
    def get_nrl_response(datalogger_keys, sensor_keys):
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
        """ Simple dispatch to determine which response function to use."""
        # if no response columns are found exit early
        if response_cols.isdisjoint(set(series.index)):
            return
        _maybe_add_nrl_response(series, channel_kwargs)
        if "response" not in channel_kwargs:
            _maybe_add_client_response(series, channel_kwargs)

    def _get_nrl_keys(series):
        """Return the sensor and datalogger keys or raise ValueError if invalid."""
        # bail out of required columns do not exist
        if not {"sensor_keys", "datalogger_keys"}.issubset(set(series.index)):
            raise ValueError("Series doesn't have required keys")
        # determine if both required columns are populated, else bail out
        sensor = series["sensor_keys"]
        datalogger = series["datalogger_keys"]
        invalid_sensor = pd.isnull(sensor) or len(sensor) == 0
        invalid_logger = pd.isnull(datalogger) or len(datalogger) == 0
        if invalid_sensor or invalid_logger:
            raise ValueError("Invalid NRL keys found in series")
        return sensor, datalogger

    def _maybe_add_client_response(series, kwargs):
        """Add the client response if a valid get_station_kwargs are found."""
        try:
            get_station_kwargs = _get_station_kwargs(series)
        except ValueError:
            return
        client = get_station_client()
        sub_inv = client.get_stations(level="response", **get_station_kwargs)
        channels = sub_inv.get_contents()["channels"]
        if not len(channels) == 1:
            msg = f"More than one channel returned by client with kwargs " f"{kwargs}"
            raise AmbiguousResponseError(msg)
        kwargs["response"] = sub_inv[0][0][0].response

    def _get_station_kwargs(series):
        """Try to get station kwargs, else raise ValueError if invalid."""
        if "get_station_kwargs" not in series.index:
            raise ValueError("No station_get_stations kwargs found")
        kwargs = series["get_station_kwargs"]
        if isinstance(kwargs, str):
            # replace single quotes with double quotes so json is valid
            try:
                return json.loads(kwargs.replace("'", '"'))
            except json.JSONDecodeError:
                raise ValueError(f"Invalid get_station_kwarg: {kwargs}")
        elif not isinstance(kwargs, dict):
            raise ValueError("No valid get_station_kwargs found")
        # convert all time columns to UTCDateTime objects
        out = dict(kwargs)  # makes a copy to not modify in place
        for time_key in set(GET_STATIONS_UTC_KWARGS) & set(out):
            out[time_key] = to_utc(out[time_key])
        return out

    def _maybe_add_nrl_response(series, channel_kwargs):
        """
        Maybe add the response information to channel kwargs if the required
        columns exist.
        """
        try:
            sensor, datalogger = _get_nrl_keys(series)
        except ValueError:
            return
        sensor_keys = _get_resp_key(sensor)
        datalogger_keys = _get_resp_key(datalogger)
        # at this point all the required info for resp lookup should be there
        channel_kwargs["response"] = get_nrl_response(datalogger_keys, sensor_keys)

    # make sure all seed_id codes are str
    for col in set(NSLC) & set(df.columns):
        df[col] = df[col].astype(str).str.replace("\.0", "")
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
