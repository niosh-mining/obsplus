"""
Module for adding a get_events method to obspy events.
"""

import inspect

import numpy as np
import obspy
import pandas as pd
from obspy.clients.fdsn import Client

import obsplus

UNSUPPORTED_PARAMS = {
    "minradius",
    "maxradius",
    "latitude",
    "longitude",
    "magnitude_type",
    "events",
    "contributor",
}
CLIENT_SUPPORTED = set(inspect.signature(Client.get_events).parameters)
SUPPORTED_PARAMS = CLIENT_SUPPORTED - UNSUPPORTED_PARAMS


def _get_ids(df, kwargs) -> set:
    """ return a set of event_ids that meet filter requirements """
    filt = np.ones(len(df)).astype(bool)

    for item, value in kwargs.items():
        if value is None:
            continue
        item = item.replace("start", "min").replace("end", "max")
        if item.startswith("min"):
            col = item.replace("min", "")
            filt &= df[col] > value
        if item.startswith("max"):
            col = item.replace("max", "")
            filt &= df[col] < value
        if item == "updatedafter":
            filt &= df["updated"] > value
        if item == "eventid":
            filt &= df["event_id"] == str(value)
    limit = kwargs.get("limit", len(df))
    return set(df.event_id[filt][:limit])


def get_events(cat: obspy.Catalog, **kwargs) -> obspy.Catalog:
    """
    Return a subset of a events filtered on input parameters.

    See obspy.core.fdsn.Client.get_events for supported arguments.

    """
    # if empty just return events
    if not kwargs:
        return cat
    # make sure all inputs are supported
    if not set(kwargs).issubset(SUPPORTED_PARAMS):
        bad_params = set(kwargs) - SUPPORTED_PARAMS
        msg = f"{bad_params} are not supported get_events parameters"
        raise TypeError(msg)
    event_ids = _get_ids(obsplus.events_to_df(cat), kwargs)
    events = [eve for eve in cat if str(eve.resource_id) in event_ids]
    return obspy.Catalog(events=events)


def get_event_summary(cat: obspy.Catalog, **kwargs) -> pd.DataFrame:
    """
    Return a dataframe from a events object after applying filters.

    See obspy.core.fdsn.Client.get_events for supported arguments.
    """
    df = obsplus.events_to_df(cat)
    event_ids = _get_ids(df, kwargs)
    return df[df.event_id.isin(event_ids)]


# --------------- monkey patch get_events method to events


obspy.Catalog.get_events = get_events
obspy.Catalog.get_event_summary = get_event_summary
