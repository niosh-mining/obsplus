"""
Module for converting obspy object to json format
"""

import json
from typing import Union

import obspy
from obspy.core.event import ResourceIdentifier, QuantityError

from obsplus.utils.events import obj_to_dict, make_class_map
from obsplus.constants import JSON_KEYS_TO_POP

JSON_SERIALIZER_VERSION = "0.0.0"  # increment when serialization changes


# -------------------- events to json functions


def cat_to_json(catalog: obspy.Catalog) -> str:
    """
    Return a json string of the catalog object
    """
    out_dict = cat_to_dict(catalog)
    return json.dumps(out_dict)


def cat_to_dict(obj):
    """
    Recursively convert cat_name object to json.

    Parameters
    ----------
    obj :
        An object that will be recursed (obspy cat_name to start)
    """
    if isinstance(obj, obspy.Catalog):
        obj.json_serializer_version = JSON_SERIALIZER_VERSION
    # if this is a non-recursible type (ie a leaf) return it
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    elif isinstance(obj, (list, tuple)):
        if not len(obj):  # if empty
            return obj
        else:  # else if list contains jsonable types
            return [cat_to_dict(x) for x in obj]
    elif isinstance(obj, dict):  # if this is a dict recurse on each value
        return {key: cat_to_dict(value) for key, value in obj.items()}
    else:  # else if this is an obspy class recurse
        return cat_to_dict(obj_to_dict(obj))


# ------------------- json to events functions


def json_to_cat(cjson: Union[str, dict]) -> obspy.Catalog:
    """
    Load json str or dict into a catalog object.

    Parameters
    ----------
    cjson
        A str or dict produced by cat_to_dict or cat_to_json.
    """
    # load json to dict
    if isinstance(cjson, str):
        cjson = json.loads(cjson)
    assert isinstance(cjson, dict)
    return obspy.Catalog(**_parse_dict_class(cjson))


dict_to_cat = json_to_cat  # alias for just passing a dict


def _parse_dict_class(cdict):
    """ parse a dictionary """
    # get intersection between cdict
    class_key = make_class_map()
    class_set = set(class_key)
    cdict_set = set(cdict)
    # get set of keys that are obspy classes in the current dict
    class_keys = class_set & cdict_set
    # iterate over keys that are also classes and recurse when needed
    for key in class_keys:
        cls = class_key[key]
        val = cdict[key]
        if isinstance(val, list):
            out = []  # a blank list for storing outputs
            for item in val:
                out.append(_init_update(item, cls))
            cdict[key] = out
        elif isinstance(val, dict):
            cdict[key] = _init_update(val, cls)
        elif isinstance(val, str):
            cdict[key] = cls(val)

    # see if any keys are resource IDs and init them (must not also be cls)
    # note: if we rely on obspy internals and v 1.1.0 we can get rid of these
    other_keys = cdict_set - class_keys
    _get_resource_ids(cdict, other_keys)
    _get_quality_errors(cdict, other_keys)
    # set of keys to remove (used by mobs) if we have an event dict
    if "origins" in cdict:
        for key in JSON_KEYS_TO_POP:
            cdict.pop(key, None)
    return cdict


def _get_resource_ids(cdict, keys):
    """Find any resource_ids and instantiate."""
    resource_id_keys = (
        key for key in keys if (key.endswith("_id") or key.endswith("_uri"))
    )
    for key in resource_id_keys:
        val = cdict[key]
        if not isinstance(val, dict):  # some _id likes are str
            continue
        cdict[key] = _init_update(val, ResourceIdentifier)


def _get_quality_errors(cdict, keys):
    """ find any quality errors and instantiate """
    quality_keys = (key for key in keys if (key.endswith("errors")))
    for key in quality_keys:
        val = cdict[key]
        cdict[key] = _init_update(val, QuantityError)


def _init_update(indict, cls):
    """ init an object from cls and update its dict with indict """
    if not indict:
        return indict
    obj = cls(**_parse_dict_class(indict))
    # some objects instantiate even with None param, fix this
    for attr in set(obj.__dict__) & set(indict):
        if indict[attr] is None:
            setattr(obj, attr, None)
    return obj


# ---------------- monkey patch json methods to events


obspy.core.event.Catalog.to_json = cat_to_json
