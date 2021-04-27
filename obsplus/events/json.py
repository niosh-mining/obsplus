"""
Module for converting obspy object to json format
"""

import json
from typing import Union, Iterable

import obspy
from obspy.core.event import Event, Catalog

import obsplus.events.schema as event_schema

JSON_SERIALIZER_VERSION = "0.0.0"  # increment when serialization changes


# -------------------- events to json functions


def _events_to_model(
    catalog: Union[Catalog, Event, Iterable[Event]]
) -> event_schema.Catalog:
    """
    Convert ObsPy events to Pydantic models.
    """
    if isinstance(catalog, Event):  # A single event was passed
        catalog = Catalog(events=[catalog])
    elif not isinstance(catalog, Catalog):  # sequence was passed
        catalog = Catalog(events=catalog)
    model = event_schema.Catalog.from_orm(catalog)
    return model


def cat_to_json(catalog: Union[Catalog, Event, Iterable[Event]]) -> str:
    """
    Convert events (catalog or event) to json string.
    """
    model = _events_to_model(catalog)
    return model.json()


def cat_to_dict(catalog: Union[Catalog, Event, Iterable[Event]]) -> dict:
    """
    Convert an event object to a
    """
    model = _events_to_model(catalog)
    return model.dict()


def dict_to_cat(cjson: Union[dict, str]) -> Catalog:
    """Convert a dictionary to a catalog"""
    if isinstance(cjson, str):
        cjson = json.loads(cjson)
    return event_schema.Catalog(**cjson).to_obspy()


json_to_cat = dict_to_cat


# ---------------- monkey patch json methods to events


obspy.core.event.Catalog.to_json = cat_to_json
