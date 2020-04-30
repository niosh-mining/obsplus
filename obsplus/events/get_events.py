"""
Module for adding a get_events method to obspy events.
"""

import inspect
from typing import Tuple, Union

import numpy as np
import obspy
import pandas as pd
from obspy.clients.fdsn import Client
from obspy.geodetics import kilometers2degrees

import obsplus
import obsplus.utils.geodetics
import obsplus.utils.misc
from obsplus.constants import get_events_parameters
from obsplus.utils.docs import compose_docstring
from obsplus.utils.time import _dict_times_to_npdatetimes

CIRCULAR_PARAMS = {"latitude", "longitude", "minradius", "maxradius", "degrees"}
NONCIRCULAR_PARAMS = {"minlongitude", "maxlongitude", "minlatitude", "maxlatitude"}

UNSUPPORTED_PARAMS = {"magnitude_type", "events", "contributor"}
CLIENT_SUPPORTED = set(inspect.signature(Client.get_events).parameters)
SUPPORTED_PARAMS = (CLIENT_SUPPORTED | CIRCULAR_PARAMS) - UNSUPPORTED_PARAMS


def _sanitize_circular_search(**kwargs) -> Tuple[dict, dict]:
    """
    Check for clashes between circular-search and box-search kwargs.

    Returns
    -------
    Two separate dictionaries of the circular kwargs and everything else.
    """
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
        if not {"minradius", "maxradius"}.intersection(kwargs):
            _ = kwargs.pop("latitude", None)
            _ = kwargs.pop("longitude", None)
    # Split parameters that are supported on sql and those that are not.
    circular_kwargs = {}
    for key in CIRCULAR_PARAMS:
        value = kwargs.pop(key, None)
        if value is not None:
            circular_kwargs.update({key: value})
    return circular_kwargs, kwargs


def _get_bounding_box(circular_kwargs: dict) -> dict:
    """
    Return a dict containing the bounding box for circular params.
    """
    circular_kwargs = dict(circular_kwargs)  # we dont want to mutate this dict
    out = {}  # init empty dict for outputs
    if "maxradius" in circular_kwargs.keys():
        maxradius = circular_kwargs["maxradius"]
        if not circular_kwargs.get("degrees", True):
            # If distance is in m we will just assume a spherical earth
            maxradius = kilometers2degrees(maxradius / 1000.0)
        # Make the approximated box a bit bigger to cope with flattening.
        out.update(
            dict(
                minlatitude=circular_kwargs["latitude"] - (1.2 * maxradius),
                maxlatitude=circular_kwargs["latitude"] + (1.2 * maxradius),
                minlongitude=circular_kwargs["longitude"] - (1.2 * maxradius),
                maxlongitude=circular_kwargs["longitude"] + (1.2 * maxradius),
            )
        )
    return out


def _get_ids(df, kwargs) -> set:
    """ return a set of event_ids that meet filter requirements """
    filt = np.ones(len(df)).astype(bool)
    # Separate kwargs used in circular searches.
    circular_kwargs, kwargs = _sanitize_circular_search(**kwargs)
    if circular_kwargs:
        # Circular kwargs are used, first apply non-circular query then trim
        kwargs.update(_get_bounding_box(circular_kwargs))
        df = get_event_summary(df, **kwargs)
        if len(df) == 0:  # If there are no events in the rectangular region.
            return set()
        filt = np.ones(len(df)).astype(bool)
        # Trim based on circular kwargs, first get distance dataframe.
        input = (circular_kwargs["latitude"], circular_kwargs["longitude"], 0)
        dist_calc = obsplus.utils.geodetics.SpatialCalculator()
        dist_df = dist_calc(input, df)
        # then get radius and filter if needed
        degrees = circular_kwargs.get("distance_degrees", True)
        radius = dist_df["distance_degrees" if degrees else "distance_m"].values
        if "minradius" in circular_kwargs:
            filt &= radius > circular_kwargs["minradius"]
        if "maxradius" in circular_kwargs:
            filt &= radius < circular_kwargs["maxradius"]
        df = df[filt]
    else:  # No circular kwargs are being used; normal query
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
        df = df[filt]
    limit = kwargs.get("limit", len(df))
    return set(df.event_id[:limit])


@compose_docstring(get_events_params=get_events_parameters)
def get_events(cat: obspy.Catalog, **kwargs) -> obspy.Catalog:
    """
    Return a subset of a events filtered on input parameters.

    Parameters
    ----------
        {get_event_parameters}
    """
    # If not kwargs are passed just return all events
    if not kwargs:
        return cat
    # Make sure all inputs are supported
    if not set(kwargs).issubset(SUPPORTED_PARAMS):
        bad_params = set(kwargs) - SUPPORTED_PARAMS
        msg = f"{bad_params} are not supported get_events parameters"
        raise TypeError(msg)
    # Ensure all times are numpy datetimes
    kwargs = _dict_times_to_npdatetimes(kwargs)
    event_ids = _get_ids(obsplus.events_to_df(cat), kwargs)
    events = [eve for eve in cat if str(eve.resource_id) in event_ids]
    return obspy.Catalog(events=events)


@compose_docstring(get_events_params=get_events_parameters)
def get_event_summary(
    cat: Union[obspy.Catalog, pd.DataFrame], **kwargs
) -> pd.DataFrame:
    """
    Return a dataframe from a events object after applying filters.

    Parameters
    ----------
        {get_event_parameters}
    """
    df = obsplus.events_to_df(cat)
    event_ids = _get_ids(df, kwargs)
    return df[df.event_id.isin(event_ids)]


# --------------- monkey patch get_events method to events


obspy.Catalog.get_events = get_events
obspy.Catalog.get_event_summary = get_event_summary
