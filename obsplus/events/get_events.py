"""
Module for adding a get_events method to obspy events.
"""

import inspect

import numpy as np
import obspy
import pandas as pd
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees

import obsplus

CIRCULAR_PARAMS = {"latitude", "longitude", "minradius", "maxradius"}

NONCIRCULAR_PARAMS = {"minlongitude", "maxlongitude", "minlatitude", "maxlatitude"}

UNSUPPORTED_PARAMS = {"magnitude_type", "events", "contributor"}
CLIENT_SUPPORTED = set(inspect.signature(Client.get_events).parameters)
SUPPORTED_PARAMS = CLIENT_SUPPORTED - UNSUPPORTED_PARAMS


def _sanitize_circular_search(**kwargs):
    if CIRCULAR_PARAMS.intersection(kwargs):
        if NONCIRCULAR_PARAMS.intersection(kwargs):
            raise ValueError(
                "{0} cannot be used with {1}".format(
                    NONCIRCULAR_PARAMS.intersection(kwargs),
                    CIRCULAR_PARAMS.intersection(kwargs),
                )
            )
        if not {"latitude", "longitude"}.issubset(kwargs):
            raise ValueError("Circular search requires both longitude and latitude")
        # If neither minradius not maxradius they just want everything.
        if "minradius" not in kwargs and "maxradius" not in kwargs:
            _ = kwargs.pop("latitude", None)
            _ = kwargs.pop("longitude", None)
    # Split parameters that are supported on sql and those that are not.
    circular_kwargs = {}
    for key in CIRCULAR_PARAMS:
        value = kwargs.pop(key, None)
        if value:
            circular_kwargs.update({key: value})
    return circular_kwargs, kwargs


def _get_ids(df, kwargs) -> set:
    """ return a set of event_ids that meet filter requirements """
    filt = np.ones(len(df)).astype(bool)

    # To speed circular searches up apply an initial box filter
    circular_kwargs, kwargs = _sanitize_circular_search(**kwargs)
    if "maxradius" in circular_kwargs.keys():
        box = {
            "minlatitude": circular_kwargs["latitude"] - circular_kwargs["maxradius"],
            "maxlatitude": circular_kwargs["latitude"] + circular_kwargs["maxradius"],
            "minlongitude": circular_kwargs["longitude"] - circular_kwargs["maxradius"],
            "maxlongitude": circular_kwargs["longitude"] + circular_kwargs["maxradius"],
        }
        for key, value in box.items():
            current_value = kwargs.get(key, None)
            if current_value is None:
                kwargs.update({key: value})
            else:
                if key.endswith("latitude"):
                    diff = abs(circular_kwargs["latitude"] - current_value)
                else:
                    diff = abs(circular_kwargs["longitude"] - current_value)
                if diff > circular_kwargs["maxradius"]:
                    kwargs.update({key: value})
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
    # Apply initial filter
    # df = set(df.event_id[filt])
    df = df[filt]
    filt = np.ones(len(df)).astype(bool)
    if circular_kwargs:
        radius = calculate_distance(
            latitude=circular_kwargs["latitude"],
            longitude=circular_kwargs["longitude"],
            df=df,
            degrees=True,
        )
        if "minradius" in circular_kwargs:
            filt &= radius > circular_kwargs["minradius"]
        if "maxradius" in circular_kwargs:
            filt &= radius < circular_kwargs["maxradius"]
        df = df[filt]
    limit = kwargs.get("limit", len(df))
    return set(df.event_id[:limit])


def calculate_distance(latitude: float, longitude: float, df, degrees=True):
    """
    Calculate the distance from all events in the dataframe to a set point.

    Parameters
    ----------
    latitude
        Latitude in degrees for point to calculate distance from
    longitude
        Longitude in degrees for point to calculate distance from
    df
        DataFrame to compute distances for. Must have columns titles
        "latitude" and "longitude"
    degrees
        Whether to return distance in degrees (default) or in kilometers.
    """

    def _dist_func(_df):
        dist, _, _ = gps2dist_azimuth(
            lat1=latitude, lon1=longitude, lat2=_df["latitude"], lon2=_df["longitude"]
        )
        if degrees:
            return kilometer2degrees(dist / 1000.0)
        return dist / 1000

    return df.apply(_dist_func, axis=1, result_type="reduce")


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
