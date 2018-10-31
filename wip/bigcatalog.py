"""
Source for the big events.
"""
import abc
import copy
from collections import defaultdict
from functools import lru_cache, partial
from types import MappingProxyType as MapProxy
from typing import Optional, Sequence, Union, Dict, Mapping, Any, List

import numpy as np
import obspy
import obspy.core.event as ev
import pandas as pd
from obspy.core.util.obspy_types import Enum

from obsplus.events.utils import make_class_map
from obsplus.utils import yield_obj_parent_attr, iterate

# dicts for mapping expected object name with class definitions and back

DTYPES = {}

# dict of functions which "flatten" non-resource id objects into a dict.
DTYPE_FUNCS = defaultdict(lambda: lambda x, y: {x: y})  # double lambda!

ATTR_TO_CLASS = make_class_map()
ATTR_TO_CLASS_NAME = {
    i: v.__name__ for i, v in ATTR_TO_CLASS.items() if hasattr(v, "__name__")
}


def _ns_to_UTC(ns):
    return obspy.UTCDateTime(ns=ns)


# {attribute_name: function to convert back to obspy object}
TO_CLASS_FUNCS = {
    i: v if not v is obspy.UTCDateTime else _ns_to_UTC for i, v in ATTR_TO_CLASS.items()
}

# ---- attributes of unusual types

BUILTINS = (int, float, str)

# define dicts to store custom type information

SIMPLE_TYPES = {obspy.UTCDateTime: int, ev.ResourceIdentifier: str, Enum: str}

TRANSFORM_FUNCS = {
    obspy.UTCDateTime: lambda x: x._ns,
    ev.ResourceIdentifier: lambda x: str(x),
    Enum: lambda x: x or "",
}

COMPLEX_TYPES = {
    ev.QuantityError: {
        "uncertainty": float,
        "lower_uncertainty": float,
        "upper_uncertainty": float,
        "confidence_level": float,
    }
}


def get_attr_recursive(obj, attrs):
    if not len(attrs):
        return obj
    else:
        out = getattr(obj, attrs[0], None)
        return get_attr_recursive(out, attrs[1:])


@lru_cache()
def _get_dtypes(cls):
    """ return a dict of attributes and datatypes for properties. """
    # handle special cases
    if cls in COMPLEX_TYPES:
        return COMPLEX_TYPES[cls]
    out = {"_parent_id_": str, "_event_id_": str}
    # get containers, properties, and types
    property_dict = getattr(cls, "_property_dict", {})
    # iterate properties, handle special case else use type
    for item, val in property_dict.items():
        # if a class that needs to be flattened
        val = type(val) if not isinstance(val, type) else val
        if hasattr(val, "_property_dict") or val in COMPLEX_TYPES:
            sub_dict = _get_dtypes(val)
            for item_, val_ in sub_dict.items():
                if item_ in {"_parent_id_", "_event_id_"}:
                    continue
                out[f"__{item}__{item_}"] = val_
        else:  # handle simple, transform types
            out[item] = SIMPLE_TYPES.get(val, val)
    # add containers (always strings refering to other tables) and return
    containers = getattr(cls, "_containers", [])
    for container in containers:
        out[f"_{container}"] = str
    return MapProxy(out)  # mapproxy to simulate immutability


# --- functions for creating tables


def _get_values(obj, parent_id=None, event_id=None):
    """ wrangle outputs into a dictionary """
    # get expected datatypes and init dict with values for sequences
    cls = type(obj) if not isinstance(obj, type) else obj
    dtypes = _get_dtypes(cls)
    out = {"_parent_id_": parent_id or "", "_event_id_": event_id or ""}
    # iterate various attrs create output dict
    for item, vtype in dtypes.items():
        if item in out:  # skip if already in out dict
            continue
        # flatten attached objects that don't have resource ids.
        if item.startswith("__"):
            attr_list = item.split("__")[1:]
            val = get_attr_recursive(obj, attr_list) or vtype()
        # this refers to another table, get the table name
        elif item.startswith("_") and item not in out:
            val = ATTR_TO_CLASS_NAME[item[1:]]
        else:
            val = getattr(obj, item, None) or vtype()
        val_type = type(val) if not isinstance(val, type) else vtype
        out[item] = TRANSFORM_FUNCS.get(val_type, lambda x: x)(val)

    return out


def _events_to_tables(
    event_iterable: Union[obspy.Catalog, Sequence[ev.Event]]
) -> Dict[str, pd.DataFrame]:
    """ Create tables from an event iterable """
    obj_dict = defaultdict(list)
    seen_ids = set()

    def _obj_to_dict(obj, parent_id=None, event_id=None):
        """ convert objects to flat dicts for conversion to pandas tables. """
        # dont process anything twice, only obj with resource_ids go in tables
        if id(obj) in seen_ids or not hasattr(obj, "resource_id"):
            return
        # get the expected datatypes and fields, then extract values into dict
        name = getattr(obj, "__name__", type(obj).__name__)
        obj_dict[name].append(_get_values(obj, parent_id, event_id))

    for event in event_iterable:
        event_id = str(event.resource_id)
        for obj, parent, attr in yield_obj_parent_attr(event):
            parent_id = getattr(parent, "resource_id", None)
            _obj_to_dict(obj, parent_id=parent_id, event_id=event_id)

    obj_dict.pop("ResourceIdentifier", None)

    tables = {i: pd.DataFrame(v).set_index("resource_id") for i, v in obj_dict.items()}
    tables["ID"] = _create_resource_id_tables(tables)
    return tables


def _create_resource_id_tables(tables):
    """ Iterate a dict of df and create a table containing """
    out = []
    for name, df in tables.items():
        dff = df[["_event_id_"]]
        dff["_table_name_"] = name
        out.append(dff)
    return pd.concat(out)


# --- tables to events


def _inflate_flattened(ser):
    """ create an object tree from flat tables. """
    # TODO make this more elegant
    out = {}
    # iterate each value, create parent objects and set attribute
    for item, value in ser.items():
        previous = None
        split = item.split("__")[1:]
        for name in split:
            # add the base class to out dict if it has not yet been created
            if previous is None and name not in out:
                previous = ATTR_TO_CLASS[name]()
                out[name] = previous
            elif previous is None and name in out:
                previous = out[name]
            # if on leaf set value on object (might be nested)
            elif name == split[-1]:
                if name in TO_CLASS_FUNCS:
                    val = TO_CLASS_FUNCS[name](value)
                else:
                    val = value or None
                setattr(previous, name, val)
            else:  # else set inited class on parent object
                new = getattr(previous, name, None)
                if new is None:  # nested class not yet inited
                    new = ATTR_TO_CLASS_NAME[name]()
                    setattr(previous, name, new)
                previous = new
    return out


def _construct_object(
    ser: pd.Series, df_dict: Dict[str, pd.DataFrame], cls, recursive=True
):
    """
    Construct the object represented by a series.

    Parameters
    ----------
    ser
    df_dict
    cls
    recursive

    Returns
    -------

    """
    # using pandas string methods to classify type of each index member
    istr = ser.index.str
    flattened_attrs = istr.startswith("__")
    nested_atts = istr.startswith("_") & (~istr.endswith("_")) & (~flattened_attrs)
    special = istr.startswith("_") & istr.endswith("_")
    basic = (~flattened_attrs) & (~nested_atts) & (~special)
    # ensure each index falls into exactly one category
    attr_sum = flattened_attrs.astype(int) + nested_atts + special + basic
    assert np.all(np.equal(attr_sum, 1))
    # put basic types into dict
    basics = ser[basic][ser[basic].astype(bool)]  # collect NonNull basic type
    out = {
        x: TO_CLASS_FUNCS[x](ser[x]) if x in TO_CLASS_FUNCS else v
        for x, v in basics.items()
    }
    # add resource id
    obj_id = ser.name
    out["resource_id"] = ev.ResourceIdentifier(obj_id)
    # inflate flattened objects
    flat = ser[flattened_attrs]
    out.update(_inflate_flattened(flat[flat.astype(bool)]))
    # add nested objects
    if recursive:
        for attr_name, table_name in ser[nested_atts].items():
            klass = ATTR_TO_CLASS[attr_name[1:]]
            # get dataframe, filter on parent id
            dff = df_dict.get(table_name, None)
            if dff is None or dff.empty:
                continue  # none of this type are defined in tables
            df = dff[dff["_parent_id_"] == obj_id]
            if df.empty:
                continue
            # recurse, creating nested objects
            func = partial(_construct_object, df_dict=df_dict, cls=klass)
            out[attr_name[1:]] = df.apply(func, axis=1).values.tolist()
    return cls(**out)


def _tables_to_catalog(df_dict: Dict[str, pd.DataFrame]) -> obspy.Catalog:
    """ convert a dict of dataframes back to an obspy events """
    try:
        events = df_dict["Event"]
    except KeyError:  # empty events
        return obspy.Catalog()
    func = partial(_construct_object, df_dict=df_dict, cls=ev.Event)
    return events.apply(func, axis=1).values.tolist()


# ---- Helper classes


class _InitEventObjectDescriptor:
    """
    A descriptor for initializing classes in obspy event heirarchy from
    instances of the appropriate class, dictionaries of kwargs, or args.

    Cleaner version of obspy.core.event.Catalog._set_resource_id and the like.
    """

    def __init__(self, cls):
        self._cls = cls

    def __set_name__(self, owner, name):
        self.name = "_" + name

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):

        if isinstance(value, Mapping):
            val = self._cls(**value)
        elif isinstance(value, Sequence):
            val = self._cls(*value)
        else:
            val = value
        setattr(instance, self.name, val)


# -------------- Engines (backends) for big events

_ENGINES = []


# --- Pandas engine


class CatalogEngine(abc.ABC):
    """ The basic interface for a bigcatalog engine. """

    # @abc.abstractmethod
    # def id_to_events(self, event_ids: [Union[str, Sequence[str]]]
    #                  ) -> List[ev.Event]:
    #     """
    #     Given one or more event ids, return the event objects.
    #
    #     Parameters
    #     ----------
    #     event_ids
    #         A sequence of resource identifiers.
    #     """

    @abc.abstractmethod
    def load_object_by_id(self, resource_id: str, recursive: bool = True) -> Any:
        """
        Load an object from the events engine using its resource identifier.

        Parameters
        ----------
        resource_id
            The ID of the object ot load
        recursive
            If True, load all child objects in the object tree as well.
        """

    @abc.abstractmethod
    def load_object_by_parent(
        self,
        parent_id: Optional[str],
        attribute: Optional[str],
        position: Optional[int] = None,
        recursive: bool = True,
    ) -> Any:
        """
        Load an object in reference to its parent.

        Requires the parent id, attribute name, and optionally, position.

        Parameters
        ----------
        parent_id
            The resource identifier of the parent object.
        attribute
            The attribute used to reference the event from the parent.
        position
            The position in a sequence of the object.
        recursive
            If True, load all child objects in the object tree as well.
        """

    def __init_subclass__(cls, **kwargs):
        """ Set docstring to abstract method on derived subclasses. """
        for name, attr in cls.__dict__.items():
            if callable(attr) and name in CatalogEngine.__dict__:
                if not getattr(cls.__dict__[name], "__doc__", None):
                    _doc = getattr(CatalogEngine.__dict__[name], "__doc__", None)
                    cls.__dict__[name].__doc__ = _doc


class PandasEngine(CatalogEngine):
    """ A bigcatalog engine that uses 'interconnected' pandas dataframes. """

    def __init__(self, events=None):
        self._dfs = _events_to_tables(events or [])
        self._event_cache = {}

    def _get_sub_tables(
        self, resource_ids: Union[str, Sequence[str]]
    ) -> Dict[str, pd.DataFrame]:
        """ return a dict of all  """
        ids = list(iterate(resource_ids))
        import pdb

        pdb.set_trace()

        return {i: v[v._event_id_.isin(ids)] for i, v in self._dfs.items()}

    def load_object_by_id(self, resource_id: str, recursive: bool = True):
        df_dict = self._get_sub_tables(resource_id)
        return _tables_to_catalog(df_dict)

    def load_object_by_parent(
        self,
        parent_id: Optional[str],
        attribute: Optional[str],
        position: Optional[int] = None,
        recursive: bool = True,
    ):
        pass

    def index_to_event(self, indicies: Sequence[int]):
        """ Return events based on index. """
        event_ids = list(self._dfs["Event"].index[indicies])
        return [self.load_object_by_id(x) for x in event_ids]


class SQLAlchemyEngine(CatalogEngine):
    pass


def start_engine(arg: Union[str, Any], **kwargs) -> CatalogEngine:
    """
    Instantiate a BigCatalog engine (backend).

    Parameters
    ----------
    arg
        Either a str that indicates engine type/parameters or a engine
        instance from SQLAlchemy.
    """
    if arg.lower() == "pandas":
        eng = PandasEngine(events=kwargs.get("events", []))
        _ENGINES.append(eng)
    else:
        raise ValueError(f"{arg} not yet supported")
    return eng


# -------- Lazy object machinery


def make_lazy(resource_id: Union[str, ev.ResourceIdentifier], engine: CatalogEngine):
    """
    Create a lazy obspy event related object.

    Lazy objects load all attributes that are built-in python types or simple
    obspy types (ie ResourceID, UTCDateTime), but defer creating more complex
    objects (which have a resource_id attribute) until needed.
    """


# ------- Big Catalog class definition


class BigCatalog:
    """
    A big events backed by Pandas dataframes.
    """

    creation_info = _InitEventObjectDescriptor(ev.CreationInfo)
    resource_id = _InitEventObjectDescriptor(ev.ResourceIdentifier)

    # --- constructors

    # standard constructor that follows events interface
    def __init__(
        self,
        events=None,
        *,
        resource_id=None,
        description=None,
        comments=None,
        creation_info=None,
        engine="pandas",
    ):
        self.events = events or []
        self.resource_id = resource_id
        self.description = description
        self.comments = comments or []
        self.creation_info = creation_info
        self._engine = start_engine(engine, events=events)

        self._dfs = _events_to_tables(events or [])
        self._event_cache = {}

    @classmethod
    def from_catalog(cls, catalog, **kwargs):
        """ Construct a BigCatalog from a standard events. """
        return cls(copy.deepcopy(catalog.events), **kwargs)

    # --- normal event interface

    def __getitem__(self, item):
        """ get catalogs from list_like structure. """
        return self._engine.index_to_event([item])[0]

    # --- dataframe methods

    def get_arrivals(
        self, event_id: Optional[str] = None, join_externals: bool = False
    ) -> pd.DataFrame:
        """
        Return dataframe of arrivals.

        Parameters
        ----------
        event_id
        join_externals

        Returns
        -------

        """
