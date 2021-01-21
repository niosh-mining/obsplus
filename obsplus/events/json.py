"""
Module for converting obspy object to json format
"""

import json
from typing import Union

import obspy
from obspy.core.event import Event, Catalog

import obsplus.events.schema as event_schema

JSON_SERIALIZER_VERSION = "0.0.0"  # increment when serialization changes


# -------------------- events to json functions


def cat_to_json(events: Union[Catalog, Event]) -> str:
    """
    Convert events (catalog or event) to json string.
    """
    if isinstance(events, Event):  # A single event was passed
        events = Catalog(events=[events])
    elif not isinstance(events, Catalog):  # sequence was passed
        events = Catalog(events=events)
    json_str = event_schema.Catalog.from_orm(events).json()
    return json_str


def cat_to_dict(events: Union[Catalog, Event]) -> dict:
    """
    Convert an event object to a
    """
    json_str = cat_to_json(events)
    return json.loads(json_str)


def dict_to_cat(cjson) -> Catalog:
    """Convert a dictionary to a catalog"""
    if isinstance(cjson, str):
        cjson = json.loads(cjson)
    return event_schema.Catalog(**cjson).to_obspy()


json_to_cat = dict_to_cat


# ---------------- monkey patch json methods to events


obspy.core.event.Catalog.to_json = cat_to_json
