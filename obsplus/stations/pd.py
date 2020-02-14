"""
Pandas functionality for stations stuff.
"""
import os
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
from obspy.core.event import Event, Catalog, WaveformStreamID
from obspy.core.inventory import Channel

import obsplus
from obsplus.constants import STATION_COLUMNS, NSLC, STATION_DTYPES
from obsplus.interfaces import BankType, EventClient
from obsplus.structures.dfextractor import (
    DataFrameExtractor,
    standard_column_transforms,
)
from obsplus.utils.misc import get_instances_from_tree, apply_to_files_or_skip

# attributes from channel to extract

stations_to_df = DataFrameExtractor(
    Channel,
    STATION_COLUMNS,
    column_funcs=standard_column_transforms,
    dtypes=STATION_DTYPES,
)


@stations_to_df.extractor()
def _extract_from_channels(channel):
    """ extract info from channels. """
    out = {x: getattr(channel, x) for x in STATION_COLUMNS[5:]}
    return out


@stations_to_df.register(obspy.Inventory)
def _extract_channel(inventory: obspy.Inventory):
    """
    Get a summary dataframe from the stations object
    """
    extras = {}
    chans = []
    for net in inventory.networks:
        for sta in net.stations:
            for chan in sta.channels:
                chan_dict = {
                    "network": net.code,
                    "station": sta.code,
                    "channel": chan.code,
                    "location": chan.location_code,
                }
                chan_dict["seed_id"] = ".".join((chan_dict[x] for x in NSLC))
                extras[id(chan)] = chan_dict
                chans.append(chan)
    return stations_to_df(chans, extras=extras)


@stations_to_df.register(str)
@stations_to_df.register(Path)
def _str_inv_to_df(path):
    """ read stations object from file or directory structure """
    path = str(path)
    # if applied to directory, recurse
    if os.path.isdir(path):
        df = pd.concat(list(apply_to_files_or_skip(_str_inv_to_df, path)))
        df.reset_index(drop=True, inplace=True)
        return df
    # else try to read single file
    try:
        return stations_to_df(obspy.read_inventory(path))
    except TypeError:
        return stations_to_df(pd.read_csv(path))


@stations_to_df.register(Event)
@stations_to_df.register(Catalog)
def _event_to_inv_df(event):
    """
    Pull WaveformStreamIDs out of an event and put it in a dataframe.
    """
    wids = get_instances_from_tree(event, WaveformStreamID)
    wid_str = {x.get_seed_string() for x in wids}
    df = pd.DataFrame(sorted(wid_str), columns=["seed_id"])
    seed = df["seed_id"].str.split(".", expand=True)
    df["network"], df["station"] = seed[0], seed[1]
    df["location"], df["channel"] = seed[2], seed[3]
    df["start_date"] = np.nan
    df["end_date"] = np.nan
    df["latitude"] = np.nan
    df["longitude"] = np.nan
    df["elevation"] = np.nan
    return stations_to_df(df)


@stations_to_df.register(BankType)
def _bank_to_df(bank):
    """ Convert the various bank types to station dataframes. """
    if isinstance(bank, EventClient):
        return stations_to_df(bank.get_events())
    if isinstance(bank, obsplus.WaveBank):
        rename = {"starttime": "start_date", "endtime": "end_date"}
        return bank.get_availability_df().rename(columns=rename)
    else:
        raise TypeError(f"{bank} type not yet supported")


@stations_to_df.register(obspy.Stream)
@stations_to_df.register(obspy.Trace)
def _stream_to_station_df(st):
    """Convert a stream/trace to station dataframe."""
    st = [st] if isinstance(st, obspy.Trace) else st
    attrs = list(NSLC) + ["starttime", "endtime"]
    stats_summary = []
    for tr in st:
        stats_summary.append({at: getattr(tr.stats, at, None) for at in attrs})
    df = pd.DataFrame(stats_summary)
    # next groupby stations and get min/max for start_date and end_date
    group = df.groupby(list(NSLC))
    df = (
        pd.concat([group["starttime"].min(), group["endtime"].max()], axis=1)
        .reset_index()
        .rename(columns={"starttime": "start_date", "endtime": "end_date"})
    )
    return stations_to_df(df)


# monkey patch in to_df method on stations


def inventory_to_dataframe(inventory_like):
    """Convert an inventory to a dataframe."""
    return stations_to_df(inventory_like)


obspy.core.inventory.Inventory.to_df = inventory_to_dataframe
