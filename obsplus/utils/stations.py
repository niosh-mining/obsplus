"""
Utilities for working with inventories.
"""
import json
import warnings
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
    SMALL_UTC,
    BIG_UTC,
    NSLC,
    STATION_DTYPES,
)
from obsplus.exceptions import AmbiguousResponseError
from obsplus.interfaces import StationClient
from obsplus.utils import get_nslc_series
from obsplus.utils.docs import compose_docstring, format_dtypes
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
station_col_str = format_dtypes(STATION_DTYPES)


class _InventoryConstructor:
    """A private helper class for constructing inventories from dataframes."""

    _client_col = "get_station_kwargs"
    _nrl_response_cols = frozenset({"datalogger_keys", "sensor_keys"})
    # define expected kwargs for inventory class constructors
    net_map = {"code": "network", "operator": "operator"}
    sta_map = {
        "code": "station",
        "equipment": "equipment",
        "creation_date": "creation_date",
        "latitude": "latitude",
        "longitude": "longitude",
        "elevation": "elevation",
    }
    cha_map = {
        "response": "response",
        "start_date": "start_date",
        "end_date": "end_date",
        "code": "channel",
        "location_code": "location",
        "sample_rate": "sample_rate",
        "sensor": "sensor",
        "equipment": "equipment",
        "elevation": "elevation",
        "azimuth": "azimuth",
        "dip": "dip",
        "data_logger": "data_logger",
        "latitude": "latitude",
        "longitude": "longitude",
        "depth": "depth",
    }
    # A set of expected columns
    _expected_cols = (
        set(net_map.values()) | set(cha_map.values()) | set(sta_map.values())
    )
    # A set of columns to drop before building inventory
    _drop_cols = list(_nrl_response_cols) + [_client_col, "index", "seed_id"]
    # columns for performing groupby at various levels
    _gb_cols = dict(
        network=("network",),
        station=("station", "latitude", "longitude", "elevation"),
        channel=("channel", "location", "start_date", "end_date"),
    )

    def __init__(self, station_client=None):
        self._client = station_client

    def _groupby_if_exists(self, df, level):
        """Groupby columns if they exist on dataframe, else return empty."""
        columns = list(self._gb_cols[level])
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
            # Pandas added an annoying warning about iterating groupbys with
            # len one. In the future ind will be a tuple, so future proofing.
            # See pandas/issues/42795
            ind = ind[0] if isinstance(ind, tuple) and len(ind) == 1 else ind
            # replace NaN values
            sub_nan = isnan.loc[df_sub.index]
            has_nan = sub_nan.any(axis=0)
            cols_with_nan = has_nan[has_nan].index
            for col in cols_with_nan:
                df_sub.loc[sub_nan[col].values, col] = np.nan
            yield ind, df_sub

    def _get_kwargs(self, series, key_mapping):
        """create the kwargs from a series and key mapping."""
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

    @property
    @lru_cache()
    def nrl_client(self):
        """Initiate a nominal response library object."""
        from obspy.clients.nrl import NRL

        return NRL()

    @property
    @lru_cache()
    def station_client(self):
        """
        Instantiate an IRIS FDSN client or return the provided client.
        """
        client = self._client
        if self._client is None:
            # importing Client can make a network call ;only do it if needed
            from obspy.clients.fdsn import Client

            client = Client()
        return client

    @lru_cache()
    def get_nrl_response(self, datalogger_keys, sensor_keys):
        nrl = self.nrl_client
        kwargs = dict(datalogger_keys=datalogger_keys, sensor_keys=sensor_keys)
        return nrl.get_response(**kwargs)

    def _get_station_client_response(self, kwargs):
        """Use the client to get station responses."""
        client = self.station_client
        sub_inv = client.get_stations(level="response", **kwargs)
        channels = sub_inv.get_contents()["channels"]
        if not len(channels) == 1:
            msg = (
                f"More than one channel returned by client with kwargs:"
                f"{kwargs}, add constraints to resolve the issue"
            )
            raise AmbiguousResponseError(msg)
        return sub_inv[0][0][0].response

    def _update_nrl_response(self, response, df):
        """Update the responses with NRL."""
        # df doesn't have needed columns, just exit
        logger_keys, sensor_keys = df["datalogger_keys"], df["sensor_keys"]
        valid = (~logger_keys.isna()) & (~sensor_keys.isna())
        response[valid] = [
            self.get_nrl_response(tuple(x), tuple(y))
            for x, y in zip(logger_keys[valid], sensor_keys[valid])
        ]

    def _update_client_responses(self, response, df):
        """Update the client responses."""
        # infer get_stations kwargs from existing columns
        provided_kwargs = df[self._client_col]
        # only try for chans that have kwargs and no response (yet)
        is_valid = (~pd.isnull(provided_kwargs)) & pd.isnull(response)
        base_kwargs = df[is_valid].apply(self.infer_get_station_kwargs, axis=1)
        for ind in base_kwargs.index:
            # get input
            input_kwargs = base_kwargs[ind]
            input_kwargs.update(provided_kwargs[ind])
            response[ind] = self._get_station_client_response(input_kwargs)

    def infer_get_station_kwargs(self, ser):
        """Infer values for get stations kwargs from a series."""
        out = {x: ser[x] for x in NSLC}
        return out

    def get_valid_json_keys(self, value):
        """
        Iterate a series, load its contents using json module if str, set
        to None if null-ish value is found.
        """
        if not value:
            value = None
        if isinstance(value, str):
            # replace single quotes with double to make json str more flexible
            value = json.loads(value.replace("'", '"'))
        return value

    def _check_only_one_response_method(self, df):
        """Raise if both response methods are specified."""
        valid_nrl_cols = ~df[self._nrl_response_cols].isnull().all(axis=1)
        valid_client_cols = ~df[self._client_col].isnull()
        both_types_used = valid_nrl_cols & valid_client_cols
        if both_types_used.any():
            bad_nslc = get_nslc_series(df[both_types_used])
            msg = (
                f"The following channels specify both a NRL and station "
                f"client response methods, choose one or the other:\n "
                f"{bad_nslc}."
            )
            raise AmbiguousResponseError(msg)

    def _load_response_columns(self, df):
        """Else raise an AmbiguousResponseError"""
        # Load any json in any of the response columns
        # Note: we copy the df at the start so mutation is ok
        for col in [self._client_col] + list(self._nrl_response_cols):
            if col not in df.columns:
                df[col] = [None] * len(df)
            else:
                df[col] = df[col].apply(self.get_valid_json_keys)
        self._check_only_one_response_method(df)
        return df

    def _get_responses(self, df):
        """Return a series of response objects."""
        # init empty series of None for storing responses
        responses = pd.Series(index=df.index, dtype=object)
        responses.loc[responses.isnull()] = None
        # early return if neither response column is found
        resp_cols = [self._client_col] + list(self._nrl_response_cols)
        if set(resp_cols).isdisjoint(set(df.columns)):
            return responses
        # Ensure both methods are not requested for any rows
        self._load_response_columns(df)
        # update responses
        self._update_nrl_response(responses, df)
        self._update_client_responses(responses, df)
        return responses

    def _add_dates(self, kwargs, some_list):
        """Add min start_date and max end_date to kwargs."""
        if not some_list:
            return
        start_list = [x.start_date or SMALL_UTC for x in some_list]
        end_list = [x.end_date or BIG_UTC for x in some_list]
        kwargs["start_date"] = np.min(start_list)
        kwargs["end_date"] = np.max(end_list)
        return kwargs

    def _maybe_warn_on_unexpected_columns(self, df):
        """Issues a UserWarning if unexpected columns are found."""
        col_set = set(df.columns)
        diff = col_set - self._expected_cols
        if diff:
            msg = f"df_to_inventory found unexpected columns: {diff}"
            warnings.warn(msg)

    def _make_inventory(self, df: pd.DataFrame):
        """
        Loopy logic for creating the inventory form a dataframe.
        """
        # get dataframe with correct columns/conditioning from input
        df = obsplus.stations_to_df(df).copy()
        # add responses (if requested) and drop response cols
        df["response"] = self._get_responses(df)
        df = df.drop(columns=self._drop_cols, errors="ignore")
        # warn if any unexpected columns are found in df
        self._maybe_warn_on_unexpected_columns(df)
        # Iterate networks and create stations
        networks = []
        for net_code, net_df in self._groupby_if_exists(df, "network"):
            stations = []
            for st_code, sta_df in self._groupby_if_exists(net_df, "station"):
                if not st_code[0]:
                    continue
                channels = []
                for ch_code, ch_df in self._groupby_if_exists(sta_df, "channel"):
                    if not ch_code[0]:  # skip empty channel lines
                        continue
                    chan_series = ch_df.iloc[0]
                    kwargs = self._get_kwargs(chan_series, self.cha_map)
                    # try to add the inventory
                    channels.append(Channel(**kwargs))
                kwargs = self._get_kwargs(sta_df.iloc[0], self.sta_map)
                self._add_dates(kwargs, channels)
                stations.append(Station(channels=channels, **kwargs))
            kwargs = self._get_kwargs(net_df.iloc[0], self.net_map)
            self._add_dates(kwargs, stations)
            networks.append(Network(stations=stations, **kwargs))

        return obspy.Inventory(
            networks=networks, source=f"ObsPlus_v{obsplus.__version__}"
        )

    __call__ = _make_inventory


@compose_docstring(station_columns=station_col_str)
def df_to_inventory(
    df: pd.DataFrame, client: Optional[StationClient] = None
) -> obspy.Inventory:
    """
    Create an inventory from a dataframe.

    Parameters
    ----------
    df
        A dataframe with the same columns and dtypes as the one returned by
         :func:`obsplus.stations_to_df`, which are:
            {station_columns}
        extra columns, except for those mentioned in the notes, are ignored.
    client
        Any client with a `get_stations` method. Only used if the dataframe
        contains special columns for retrieving channel responses.

    Notes
    -----
    There are two ways to include response information:

    1. Using the Nominal Response Library (NRL):
    (https://docs.obspy.org/master/packages/obspy.clients.nrl.html)
    If the dataframe has columns named "sensor_keys" and "datalogger_keys"
    these will indicate the response information should be fetched using
    ObsPy's NRL client. Each of these columns should either contain lists
    or josn-loadable strings.
    For example, to specify sensor keys:
    ["Nanometrics", "Trillium 120 Horizon"]
    or
    '["Nanometrics", "Trillium 120 Horizon"]'
    are both valid.

    2. Using a station client:
    If the dataframe contains a column get_stations_kwargs it indicates that
    either a client was passed as the client argument or the fdsn IRIS client
    should be used to download at least *some* channel response information.
    The contents of this column must be a dictionary or json string of
    acceptable keyword arguments for the client's `get_stations` method.
    All time values must be provided as iso8601 strings.
    For example,
    {'network': 'UU', 'station': 'TMU', 'location': '01', 'channel': 'HHZ',
     'starttime': '2017-01-02',}
    would be passed to the provided client to download a response for the
    corresponding channel.
    - If more than one channel is returned from the get_stations call an
      AmbiguousResponseError will be raised and a more specific query will
      be required.
    - If not all the required seed_id information is provided it will be
      ascertained from the appropriate column.
    - To simply fetch a response using only the info provided in other columns
      use an empty dict, or json string (eg '{}').
    - No responses will be fetched for any rows with empty strings or null
      values in the get_stations_kwargs column.
    - If both NRL and client methods are indicated by column contents an
      AmbiguousResponseError will be raised.
    """
    inv_constructor = _InventoryConstructor(station_client=client)
    return inv_constructor(df)


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
