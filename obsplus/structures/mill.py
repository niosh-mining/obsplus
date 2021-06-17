"""
Module for converting tree-like structures into tables and visa-versa.
"""
import copy
from collections import defaultdict
from functools import lru_cache
from typing import Type, Dict, Optional, Tuple, Sequence

import obsplus
import pandas as pd
from obsplus.structures.model import ObsPlusModel
from obsplus.exceptions import IncompatibleDataFramesError


class Mill:
    """
    A class for managing tree-like data structures with table slices.

    Currently this just uses instances of ObsPlusModels but we plan to
    switch to awkward array in the future.
    """

    _model: Type[ObsPlusModel] = ObsPlusModel
    _id_name: Optional[str] = None
    _id_map: Dict[str, tuple]
    _data: dict
    _dataframers: Dict[str, Type["obsplus.DataFramer"]]
    _type_map_key: str = "__id_type__"

    def __init_subclass__(cls, **kwargs):
        """Ensure subclasses have their own framers dict."""
        cls._dataframers = {}

    def __init__(self, data):
        self._data = data

    @property
    @lru_cache()
    def _df_dicts(self):
        df_dicts = _dict_to_tables(
            data=self._get_data(self._data),
            master_schema=self._model.get_obsplus_schema(),
            data_type=self._model.__name__,
        )
        df_dicts = self._post_df_dict(df_dicts)
        df_dicts["__summary__"] = self._add_df_summary(df_dicts)
        return df_dicts

    def _get_data(self, data) -> dict:
        """Get the internal data structure."""
        if isinstance(data, dict):
            return data
        else:
            return self._model.from_orm(data).dict()

    @classmethod
    def register_data_framer(cls, name):
        """
        Register a dataframer on this mill.
        """

        def _func(framer: "obsplus.DataFramer"):
            cls._dataframers[name] = framer
            return framer

        # TODO add check for already defined mappers
        return _func

    def get_df(self, name):
        """
        Return a dataframe generated by a registered DataFramer.

        Parameters
        ----------
        name
            A string identifying the dataframe.
        """
        try:
            Framer = self._dataframers[name]
        except KeyError:
            msg = (
                f"Unknown dataframer {name}, support framers are: \n"
                f"{list(self._dataframers)}"
            )
            raise KeyError(msg)
        framer = Framer(self)
        return framer.get_dataframe(self._data, stash=self._stash)

    def get_referred_address(self, id_str) -> Tuple:
        """
        Get the address of the requested ID string.

        An empty tuple is returned if it is not found.
        """
        if id_str in self._id_map:
            return self._id_map[id_str]
        self._id_map = self._index_resource_ids()
        return self._id_map.get(id_str, ())

    def __str__(self):
        name = self._model.__name__
        obj_count = len(self._df_dicts[self._type_map_key])
        msg = f"Mill with spec of [{name}] and [{obj_count}] managed objects"
        return msg

    def lookup(self, ids: Sequence[str]) -> pd.DataFrame:
        """
        Look up rows from a dataframe which have ids.

        Parameters
        ----------
        ids
            A sequence of Ids.

        Raises
        ------
        IncompatibleDataFramesError
            If the ids belong to different types of dataframes.
        KeyError
            If a requested ID is not contained by the mill.

        Notes
        -----
        - If ids do not exist in Mill an empty df is returned
        """
        if isinstance(ids, str):
            ids = [ids]
        dtypes = self._df_dicts[self._type_map_key].loc[ids].unique()
        if len(dtypes) > 1:
            msg = "Provided ids belong to multiple types of objects"
            raise IncompatibleDataFramesError(msg)
        elif not len(dtypes):  # no dataframe found
            return pd.DataFrame()

        new_table = self._df_dicts[dtypes.unique().astype(str)[0]]
        return new_table.loc[ids]



    # methods for custom df_dict creations
    def _add_df_summary(self, df_dicts):
        """Add a summary table to the dataframe dict for easy querying."""
        return {}

    def _post_df_dict(self, df_dicts):
        """This can be subclassed to run validators/checks on dataframes."""
        return df_dicts



def _dict_to_tables(
    data,
    master_schema,
    data_type: str,
    id_field: str = "resource_id",
    scope_type=None,
) -> Dict[str, pd.DataFrame]:
    """
    Decompose a json-like data-structure with known schema into tables.

    Parameters
    ----------
    data
        A list or dict with nested json supported objects.
    master_schema
        An obsplus schema which defines the structure of data.
        See :meth:`obsplus.structures.model.Model.get_obsplus_schema`
    data_type
        The key in schema which describes the data's type.
    id_field
        The field used as a unique identifier.
    scope_type
        The name of the class which will scope remaining data (eg events)

    Notes
    -----
    This is not fully generic; there are certain patterns (eg
    heterogeneous arrays) which are not supported.

    Returns
    -------
    A dict of {classname: df}
    """
    index_columns = [id_field, "parent_id", "scope_id", "index", "attr"]

    def _flatten_ids(obj, id_attrs):
        """Flatten ids from dict of {id: id_str} to just id_str."""
        for key in id_attrs:
            if isinstance(obj[key], dict):
                obj[key] = obj[key].get("id")
        return obj

    def _recurse(
        obj: dict,
        current_type: str,
        parent_id=None,
        scope_id=None,
        attr=None,
        index=None,
    ):
        """handles dictionary decomposition."""
        schema_ = master_schema[current_type]
        obj = copy.copy(obj)  # make a shallow copy of dict as we do modify it
        # flatten all resource_ids
        _flatten_ids(obj, schema_["id_attrs"])
        current_id = obj.get(id_field)
        if current_type == scope_type:
            scope_id = current_id
        # get set of keys which are sub models
        for sub_key in set(schema_["attr_ref"]) - set(schema_["id_attrs"]):
            sub_obj = obj.pop(sub_key)
            if not sub_obj:  # skip empty items
                continue
            new_type = schema_["attr_ref"][sub_key]
            # handle sub dicts
            if not isinstance(sub_obj, list):
                _recurse(sub_obj, new_type, current_id, scope_id, sub_key)
                continue
            # handle sub arrays
            for num, sub in enumerate(sub_obj):
                _recurse(sub, new_type, current_id, scope_id, sub_key, num)
        # add indices
        obj.update(
            {
                id_field: current_id,
                "parent_id": parent_id,
                "scope_id": scope_id,
                "attr": attr,
                "index": index,
            }
        )
        object_lists[current_type].append(obj)
        return obj

    def _make_df_dict(object_list):
        """Convert a dist of lists of dicts into a dict of DFs."""
        out = {}
        table_classes = set(master_schema) - {"ResourceIdentifier"}
        for dtype in table_classes:
            if dtype.startswith("ResourceIdentifier"):
                continue
            schema_ = master_schema[dtype]
            # get columns which are resource ids or basic types
            _not_referred = set(schema_["attr_type"]) - set(schema_["attr_ref"])
            cols = sorted((_not_referred | schema_["id_attrs"]) - {id_field})
            df_list = object_list[dtype]
            if not df_list:
                all_cols = sorted(cols + index_columns)
                df = pd.DataFrame(columns=all_cols).set_index(index_columns)
            else:
                df = pd.DataFrame(df_list).set_index(index_columns)[cols]
            out[dtype] = df.sort_index()
        out["__id_type__"] = _make_id_type_lookup(out)
        return out

    def _make_id_type_lookup(_df_dict):
        """Make a series of resource_id: type"""
        dtypes_df = []

        categorical = pd.CategoricalDtype(list(master_schema))
        for i, v in _df_dict.items():
            if not i.startswith("__"):
                dtypes_df.append(pd.DataFrame(i, index=v.index, columns=["dtype"]))
        out = pd.concat(dtypes_df).astype(categorical)
        return out["dtype"]

    # populate dicts of lists for each type.
    object_lists = defaultdict(list)
    _recurse(data, data_type)
    # convert list of dicts to dataframes
    out = _make_df_dict(object_lists)
    return out
