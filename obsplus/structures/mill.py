"""
Module for converting tree-like structures into tables and visa-versa.
"""
import copy
import json
import uuid
from collections import defaultdict
from contextlib import suppress
from functools import wraps
from pathlib import Path
from typing import Set, Union
from typing import Type, Dict, Optional, Tuple, Sequence, TypeVar

import numpy as np
import pandas as pd
from typing_extensions import Annotated, get_origin

import obsplus
from obsplus.constants import SUPPORTED_MODEL_OPS
from obsplus.exceptions import IncompatibleDataFramesError, InvalidModelAttribute
from obsplus.structures.model import ObsPlusModel, FunctionCall
from obsplus.utils.misc import (
    argisin,
    register_func,
    make_zip_archive,
    extract_zip_archive,
    property_cache,
)
from obsplus.utils.pd import (
    cast_dtypes,
    get_index_group,
    int64_to_int_obj,
    index_intersect,
)

MillType = TypeVar("MillType", bound="Mill")


def _inplace_or_copy(func):
    """Decorator for performing a function inplace or copying self."""

    @wraps(func)
    def _func(self, *args, **kwargs):
        inplace = kwargs.get("inplace", False)
        if not inplace:
            kwargs["inplace"] = True
            new = self.copy()
            new_func = getattr(new, func.__name__)
            return new_func(*args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return _func


class Mill:
    """
    A class for turning tree-like data structures into table slices.
    """

    model: Type[ObsPlusModel] = ObsPlusModel
    _id_name: Optional[str] = None
    _data: dict
    _dataframers: Dict[str, Type["obsplus.DataFramer"]]
    _structure_key: str = "__structure__"
    _df_dicts: Dict[str, pd.DataFrame] = None
    _scope_model_name = None
    _meta_dicts: Dict[str, pd.DataFrame] = None
    _schema: Optional[dict] = None
    _schema_file_name = ".schema.json"
    _meta_file_name = ".meta.json"
    _id_cls_name = "ResourceIdentifier"
    _rid_string = "resource_identifier"
    __version__ = "0.0.0"

    def __init_subclass__(cls, **kwargs):
        """Ensure subclasses have their own framers dict."""
        cls._dataframers = {}

    def __init__(self, data):
        self._data = self._get_data(data)

    @classmethod
    def _from_df_dict(cls, df_dict, meta=None, schema=None) -> MillType:
        """init instance from df dict."""
        out = cls(None)
        out._df_dicts = df_dict
        out._meta = meta
        out._raw_schema = schema
        return out

    @property
    @property_cache
    def schema(self):
        """A reshaped version of raw schema for translating data to/from df."""
        return _get_schema_dicts(self.df_schema)

    @property
    @property_cache
    def raw_schema(self):
        """Simply returns the json schemas provided by the pydantic model."""
        return self.model.schema()

    @property
    @property_cache
    def df_schema(self):
        """Simply returns the json schemas provided by the pydantic model."""
        return _get_schema_df(self.raw_schema, self._id_cls_name)

    def _get_table_dict(self):
        if self._df_dicts is not None:
            return self._df_dicts
        df_dicts = _dict_to_tables(
            data=self._get_data(self._data),
            schema=self.schema,
            data_type=self.model.__name__,
            scope_type=self._scope_model_name,
        )
        df_dicts = self._post_df_dict_hook(df_dicts)
        self._df_dicts = df_dicts
        df_dicts["__summary__"] = self._get_summary(df_dicts)
        return df_dicts

    @property
    def structure_df(self):
        """Get schema from model."""
        return self._get_table_dict()["__structure__"]

    def _get_data(self, data) -> dict:
        """Get the internal data structure."""
        if isinstance(data, dict):
            return data
        else:
            return self.model.from_orm(data).dict()

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

    def get_df(self, name, **kwargs):
        """
        Return a dataframe generated by a registered DataFramer.

        Parameters
        ----------
        name
            A string identifying the dataframe.
        kwargs
            Any kwargs supported by scope query.
        """
        # first look for native names to mill
        if name in self._get_table_dict():
            df = self._get_table_dict()[name]
            if kwargs:
                scope_ids = self.get_scope_ids(**kwargs)
                df = index_intersect(df, scope_ids)
            return df
        try:
            Framer = self._dataframers[name]
        except KeyError:
            msg = (
                f"Unknown dataframe: {name}, known dataframes are: \n"
                f"{list(self._dataframers) + list(self._get_table_dict())}"
            )
            raise KeyError(msg)
        framer = Framer()
        df = self._get_df_from_framer(framer, **kwargs)

        return df

    def get_summary_df(self):
        """Return a dataframe which summarizes objects on defined level."""
        return self.get_df("__summary__")

    def _get_df_from_framer(self, framer, **kwargs):
        scope_ids = self.get_scope_ids(**kwargs) if kwargs else None
        out = {}
        base_name = framer._model_name
        for attr_name, tracker in framer._fields.items():
            specs = tracker.spec_tuple
            dtype = framer._dtypes.get(attr_name)
            resolver = _OperationResolver(
                self, specs, base_name, dtype, scope_ids=scope_ids
            )
            out[attr_name] = resolver()
        df = pd.DataFrame(out)
        return df

    def __str__(self):
        cls_name = self.__class__.__name__
        model_name = self.model.__name__
        obj_count = len(self._get_table_dict()[self._structure_key])
        msg = (
            f"{cls_name} with spec of [{model_name}] and [{obj_count}] managed objects"
        )
        return msg

    __repr__ = __str__

    def lookup(self, ids: Sequence[str]) -> Tuple[pd.DataFrame, str]:
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
        df = self._get_table_dict()[self._structure_key].loc[ids]
        models = df["model"].unique()
        if len(models) > 1:
            msg = "Provided ids belong to multiple types of objects"
            raise IncompatibleDataFramesError(msg)
        elif not len(models):  # no objects with given ids
            return pd.DataFrame(), pd.NA
        model_name = models[0]
        new_table = self._get_table_dict()[model_name]
        return new_table.loc[ids], model_name

    def get_children(
        self,
        cls,
        attr_name,
        df=None,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Find all children of cls contained in attr_name.

        Parameters
        ----------
        cls
            The class name of the data.
        attr_name
            The attr name to search for children.
        df
            A dataframe of current parents. If none, simply use all
            rows in cls table.

        Examples
        --------
        # get all picks on events
        >>> import obsplus
        >>> cat = obsplus.load_dataset('bingham_test').event_client.get_events()
        >>> mill = obsplus.EventMill(cat)  # init EventMill
        >>> pick_df, child_model = mill.get_children('Event', 'picks')
        """
        if df is None:
            df = self.get_df(cls)
        try:
            model_name = self.schema["models"][cls][attr_name]
        except KeyError:
            msg = f"{cls} has no model attributes of {attr_name}"
            raise InvalidModelAttribute(msg)
        else:
            child_table = self.get_df(model_name)
        structure = self.structure_df

        child_structure = structure.loc[child_table.index]
        is_valid = child_structure["parent_id"].isin(df.index)

        return child_table.loc[is_valid[is_valid].index], model_name

    def get_parent_ids(
        self,
        ids: Sequence[str],
        level: Optional[int] = None,
        targets: Optional[Sequence[str]] = None,
    ) -> pd.Series:
        """
        Transverse the object tree up to get parent ids of provided ids.

        Parameters
        ----------
        ids
            The ids for which parent ids will be found
        level
            A maximum number of levels to transverse up the object tree.
            If None, go until the first orphan node is found.
        targets
            A set of expected ids when reached tree transversal is halted.
        """
        sdf = self.get_df("__structure__")
        max_iterations = 1_000
        targets = set() if targets is None else set(targets)
        orphans = set()

        out = pd.Series(list(ids), index=list(ids))
        for num in range(max_iterations):
            # determine if level requirements have been met
            if level is not None and num >= level:
                break
            not_orphaned = ~out.isin(orphans)
            unmet_target = ~out.isin(targets)
            sub_ser = out[not_orphaned & unmet_target]
            if not len(sub_ser):  # all termination conditions met
                break
            parents = sdf.loc[sub_ser.values, "parent_id"]
            # determine if any orphans are found, update orphan set and
            # set orphans to be there own parents (who else will do it?)
            is_orphan = parents.isnull().values
            if np.any(is_orphan):
                orphans |= set(parents[is_orphan].index)
                parents[is_orphan] = parents[is_orphan].index.values
            parents.index = sub_ser.index
            out.update(parents)
        return out

    # methods for custom df_dict creations
    def _get_summary(self, df_dicts):
        """Add a summary table to the dataframe dict for easy querying."""
        return pd.DataFrame()

    def get_structure_df(self, **kwargs):
        """
        Get the df defining object structure.

        Optionally, filter structure based on scope_ids.
        """
        df = self._get_table_dict()["__structure__"]
        if kwargs:
            df = df.loc[self.get_scope_ids(**kwargs)]
        return df

    def _post_df_dict_hook(self, df_dicts):
        """This can be subclassed to run validators/checks on dataframes."""
        return df_dicts

    def to_model(self, **kwargs):
        """Return a populated model with data from mill."""
        schema = self.schema
        data = _tables_to_dict(
            table_dict=self._get_table_dict(),
            cls_name=self.model.__name__,
            schema=schema,
            structure_df=self.get_structure_df(**kwargs),
        )
        return self.model(**data)

    def get_scope_ids(self, **kwargs) -> Optional[Set[str]]:
        """
        Get scope ids given a query.

        This only works if the summary dataframe is defined.
        """
        return None

    def copy(self, deep=False):
        """Return a copy of the mill."""
        return self.__class__._from_df_dict(copy.deepcopy(self._get_table_dict()))

    def get_meta_dict(self):
        """Write meta json file."""
        out = {
            "version": self.__version__,
        }
        return out

    def to_parquet(self, path: Union[str, Path], write_schema=True):
        """
        Save the mill to a zipped directory of parquet files.
        """
        func = pd.DataFrame.to_parquet
        path = Path(path)
        with make_zip_archive(path) as temp_path:
            data_path = temp_path / "data"
            data_path.mkdir(exist_ok=True, parents=True)
            # save data files and schemas
            for name, df in self._get_table_dict().items():
                if not len(df):
                    continue
                func(df, data_path / name)
            # write schema and meta files
            with (temp_path / self._schema_file_name).open("w") as fi:
                fi.write(self.model.schema_json())
            with (temp_path / self._meta_file_name).open("w") as fi:
                json.dump(self.get_meta_dict(), fi)

    @classmethod
    def from_parquet(cls: Type[MillType], path: Union[str, Path]) -> MillType:
        """
        Read a parquet archive into memory.

        Parameters
        ----------
        path
            The path to the parquet directory.
        """
        func = pd.read_parquet
        with extract_zip_archive(path) as temp_path:
            # extract zip file
            data = {}
            for path in (temp_path / "data").rglob("*"):
                data[path.name] = func(path)

            schema_path = temp_path / cls._schema_file_name
            meta_path = temp_path / cls._meta_file_name
            with open(schema_path, "r") as sf, open(meta_path, "r") as mf:
                meta = json.load(mf)
                schema = json.load(sf)
        out = cls._from_df_dict(df_dict=data, meta=meta, schema=schema)
        return out


def _get_schema_dicts(df_schema_dict):
    """Parses out dicts from dataframes for use in decomposition"""
    rid_dict, model_dict, dtype_dict, array_dict = {}, {}, {}, {}
    for name, df in df_schema_dict.items():
        if name.startswith("__"):
            continue
        # get set of attrs which are resource ids
        is_rid = ~df["referenced_model"].isnull()
        rid_dict[name] = set(df[is_rid].index)
        # get referenced model type
        has_model = ~(df["model"].isnull() | is_rid)
        model_dict[name] = dict(df[has_model]["model"])
        # dtype
        has_dtype = ~df["dtype"].isnull()
        dtype_dict[name] = dict(df[has_dtype]["dtype"])
        array_dict[name] = set(df.index[df["is_list"]])

    out = {
        "resource_ids": rid_dict,
        "models": model_dict,
        "dtypes": dtype_dict,
        "arrays": array_dict,
        "model_names": set(df_schema_dict),
    }
    return out


# def _get_schema_dicts(df_schema_dict):
#     """Parses out dicts from dataframes for use in decomposition"""
#     rid_dict, model_dict, dtype_dict, array_dict = {}, {}, {}, {}
#     for name, df in df_schema_dict.items():
#         if name.startswith("__"):
#             continue
#         # get set of attrs which are resource ids
#         is_rid = ~df["referenced_model"].isnull()
#         rid_dict[name] = set(df[is_rid].index)
#         breakpoint()
#         # get referenced model type
#         has_model = ~(df["model"].isnull() | is_rid)
#         model_dict[name] = dict(df[has_model]["model"])
#         # dtype
#         has_dtype = ~df["dtype"].isnull()
#         dtype_dict[name] = dict(df[has_dtype]["dtype"])
#         array_dict[name] = set(df.index[df["is_list"]])
#
#     out = {
#         "resource_ids": rid_dict,
#         "models": model_dict,
#         "dtypes": dtype_dict,
#         "arrays": array_dict,
#         "model_names": set(df_schema_dict),
#     }
#     return out


def _dict_to_tables(
    data,
    schema,
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
    schema
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
        obj = copy.copy(obj)  # make a shallow copy of dict as we do modify it
        # flatten all resource_idsschema_["id_attrs"]
        rid_attrs_set = rid_dict[current_type]
        model_attr_dict = model_dict[current_type]
        _flatten_ids(obj, rid_attrs_set)
        if id_field not in obj:
            obj[id_field] = str(uuid.uuid4())
        current_id = obj[id_field]
        if current_type == scope_type:
            scope_id = current_id
        # get set of keys which are sub models
        for sub_key in set(model_attr_dict) - rid_attrs_set:
            sub_obj = obj.pop(sub_key)
            if not sub_obj:  # skip empty items
                continue
            new_type = model_attr_dict[sub_key]
            # handle sub dicts
            if not isinstance(sub_obj, list):
                _recurse(sub_obj, new_type, current_id, scope_id, sub_key)
                continue
            # handle sub arrays
            for num, sub in enumerate(sub_obj):
                _recurse(sub, new_type, current_id, scope_id, sub_key, num)
        # add indices
        structure = {
            id_field: current_id,
            "parent_id": parent_id,
            "scope_id": scope_id,
            "attr": attr,
            "index": index,
            "model": current_type,
        }
        object_lists[current_type].append(obj)
        structure_list.append(structure)
        return obj

    def _make_structure_df():
        """Make a the structure dataframe"""
        structure = pd.DataFrame(structure_list).set_index(id_field)
        return structure

    def _make_df_dict(object_list):
        """Convert a dict of lists of dicts into a dict of DFs."""
        out = {}

        for model in schema["model_names"]:
            data = object_list.get(model, {})
            dtypes = dtype_dict[model]
            columns = sorted(set([id_field] + list(dtypes)))
            dff = (
                pd.DataFrame(data, columns=columns)
                .pipe(cast_dtypes, dtypes)
                .set_index(id_field)
            )
            out[model] = dff

        out["__structure__"] = _make_structure_df()
        return out

    dtype_dict = schema["dtypes"]
    rid_dict = schema["resource_ids"]
    model_dict = schema["models"]
    # populate dicts of lists for each type.
    object_lists = defaultdict(list)
    structure_list = []  # a list for storing structural info
    _recurse(data, data_type)
    # convert list of dicts to dataframes
    out = _make_df_dict(object_lists)
    return out


def _tables_to_dict(
    table_dict,
    cls_name,
    schema,
    structure_df,
    id_field="resource_id",
) -> dict:
    """
    Convert the table_dict back to json like structures.

    This is a very naive, loopy implementation.

    Parameters
    ----------
    table_dict
        A dict of dataframes created by :func:`_dict_to_tables`.
    """

    def _get_processed_df(table_dict, struct):
        """Return a dict of {(parent_id, attr): [object]}"""
        out = defaultdict(list)
        rid_2_parent = struct["parent_id"]
        rid_2_attr = struct["attr"]
        for name, df in table_dict.items():
            if name.startswith("_"):  # skip private df, like __structure__
                continue
            # sort rows so that objects are reassembled correctly
            sort_cols = ["parent_id", "attr", "index"]
            sub_struct = index_intersect(struct, df).sort_values(sort_cols)
            df = df.loc[sub_struct.index]
            # get values that should be None
            sub = (
                df.pipe(int64_to_int_obj)
                .astype(object)
                .where(~df.isnull(), None)
                .reset_index()
            )
            pid = sub[id_field].map(rid_2_parent)
            attr = sub[id_field].map(rid_2_attr)
            for (p, a), tup in zip(zip(pid, attr), sub.itertuples(index=False)):
                out[(p, a)].append(tup._asdict())
        return out

    def _get_attrs(data, cls_name, obj_dict):
        """Get data from attributes in data."""
        # determine attrs which are models
        attrs = set(models[cls_name]) - rids[cls_name]
        for attr in attrs:
            sub_list = obj_dict[(data[id_field], attr)]
            sub_cls_name = models[cls_name][attr]
            sub_data_list = [_get_attrs(x, sub_cls_name, obj_dict) for x in sub_list]
            # assign arrays back
            if attr in arrays[cls_name]:  # this attr should be a list
                data[attr] = sub_data_list
            else:  # this should be a single instance
                # TODO add check for optional rather than just assuming. Else skip
                data[attr] = None if not len(sub_data_list) else sub_data_list[0]
        return data

    base_df = table_dict[cls_name]
    struct = structure_df
    models, rids = schema["models"], schema["resource_ids"]
    arrays = schema["arrays"]
    assert len(base_df) == 1, "base dataframe should have length of one"
    first = dict(base_df.reset_index().iloc[0])
    processed = _get_processed_df(table_dict, struct)
    out = _get_attrs(first, cls_name, processed)
    return out


def _get_schema_df(schema: dict, rid_str: str) -> Dict[str, pd.DataFrame]:
    """ "
    Convert schema pydnatic json schema to dict of dataframes.
    """
    dtypes = {
        "model": "string",
        "dtype": "string",
        "referenced_model": "string",
        "required": "boolean",
        "is_list": "boolean",
    }

    def _get_dtype(attr_dict):
        """Get the type of the attr dict."""
        type_ = attr_dict.get("type")
        out = attr_dict.get("type", None)
        if type_ == "number":
            out = "float64"
        elif type_ == "integer":
            out = "Int64"
        # check if this is a datetime
        elif attr_dict.get("format") == "date-time":
            out = "datetime64[ns]"
        elif "anyOf" in attr_dict:
            values = [x["const"] for x in attr_dict["anyOf"]]
            out = pd.CategoricalDtype(values)
        elif out == "array":  # no array type
            out = None
        return out

    def _get_series(attr_dict):
        """
        Determine the type and reference type of an attribute from a schema dict.
        """
        ref_str = "#/definitions/"
        atype = attr_dict.get("type", "object")
        is_array = atype == "array"
        if is_array:  # dealing with array, find out sub-type
            items = attr_dict.get("items", {})
            ref = items.get("$ref", None)
        else:
            ref = attr_dict.get("$ref", None)
        if ref:  # there is a reference to a def, just get name
            ref = ref.replace(ref_str, "")

        is_rid = ref is not None and ref.startswith(rid_str)
        ref_mod = ref.replace("ResourceIdentifier", "") if is_rid else None

        out = {
            "model": ref if not is_rid else None,
            "dtype": _get_dtype(attr_dict) if not is_rid else "string",
            "referenced_model": ref_mod,
            "is_list": is_array,
        }
        return out

    def _get_dataframe(base):
        """Create dataframe from class attributes."""
        out = {}
        required = base.get("required", {})
        for name, prop in base["properties"].items():
            out[name] = _get_series(prop)
            out[name]["required"] = name in required
        return pd.DataFrame(out).T.astype(dtypes)[list(dtypes)]

    out = {
        schema["title"]: _get_dataframe(
            schema,
        )
    }
    for name, sub_schema in schema["definitions"].items():
        if name.startswith(rid_str):
            continue
        out[name] = _get_dataframe(
            sub_schema,
        )
    return out


class _OperationResolver:
    """
    A class for resolving operations specifying data access from models.
    """

    fill_types = {
        str: "",
        "str": "",
    }
    _funcs = {}

    def __init__(self, mill: Mill, spec, model_name, dtype, scope_ids=None):
        self.mill = mill
        self.spec = spec if isinstance(spec, tuple) else spec.spec_tuple
        self._dtype = self._get_dtype(dtype)
        self._struct_df = mill.get_structure_df()
        base_df = mill.get_df(model_name)
        if scope_ids is not None:
            base_df = index_intersect(base_df, scope_ids)
            self._struct_df = self._struct_df.loc[scope_ids]
        # get base (starting place) info
        self._base_df = base_df
        self._base_cls = model_name
        # set attrs which update for each operation
        self.current_ = base_df
        self.cls_ = model_name
        # maps the id of current back to base objects
        self.base_id_map_ = pd.Series(base_df.index, index=base_df.index)
        self.scope_ids = scope_ids

    @register_func(_funcs, "aggregate")
    def _aggregate(self, func):
        """Perform an aggregation on a column using a provided function."""
        df = self.current_.copy()
        df.index = self.base_id_map_.index
        out = df.groupby(level=0).apply(func)
        inds = out.index.values
        self.current_ = out
        self.base_id_map_ = pd.Series(inds, index=inds)

    @register_func(_funcs, "parent")
    def _get_parent(
        self,
    ):
        """Get the parent of the current rows, expanding if needed."""
        parents_id_map = self._struct_df.loc[self.current_.index, "parent_id"]
        parents, cls = self.mill.lookup(np.unique(parents_id_map.values))
        self.base_id_map_ = self.base_id_map_.map(parents_id_map)
        self.current_ = parents
        self.cls_ = cls

    @register_func(_funcs, "match")
    def _get_match(self, _invers=False, **kwargs):
        """Match columns, returned filtered df."""
        df = self.current_
        out = pd.Series(np.ones(len(df)), index=df.index).astype(bool)
        for colname, condition in kwargs.items():
            ser = df[colname]
            out &= ser.str.match(condition)
        if _invers:
            out = ~out
        self._reduce_current(df[out])

    @register_func(_funcs, "antimatch")
    def _get_antimatch(self, **kwargs):
        """Perform antimatch (inversed match)."""
        self._get_match(_invers=True, **kwargs)

    @register_func(_funcs, "last")
    def _get_last(self):
        """return the last of each item (by index, attr, parent_id)."""
        df = self.current_
        struct = self._struct_df.loc[df.index]
        last_ids = struct.groupby(["parent_id", "attr"])["index"].idxmax()
        self._reduce_current(df.loc[last_ids.values])

    @register_func(_funcs, "first")
    def _get_first(self):
        """return the first of each item (by index, attr, parent_id)."""
        df = self.current_
        struct = self._struct_df.loc[df.index]
        last_ids = struct.groupby(["parent_id", "attr"])["index"].idxmin()
        self._reduce_current(df.loc[last_ids.values])

    def _reduce_current(self, df):
        """
        Update class state (current_ and base_id_map_) using df which
        has a subset of columns from current_
        """
        bid = self.base_id_map_
        args = argisin(df.index.values, bid.values)
        self.base_id_map_ = pd.Series(df.index.values, index=bid.index.values[args])
        self.current_ = df

    def __call__(self, **kwargs):
        """Resolve operations on mill."""
        # apply all operation
        for op in self.spec[1:]:
            self.dispatch(op)
            # ensure base_id map stays updated
            assert set(self.base_id_map_.values) == set(self.current_.index.values)
            if len(self.current_) == 0:  # once we hit an empty state bail out
                self.current_ = pd.Series(name=self.spec[-1], dtype=float)
                break
        # join result to base
        assert self.base_id_map_.index.unique().all(), "index must now be unique"
        last = self.current_
        assert isinstance(last, pd.Series)
        # remap index back to original ids
        re_map = last.loc[self.base_id_map_.values]
        re_map.index = self.base_id_map_.index
        dtype = {last.name: self._dtype} if self._dtype is not None else {}
        out = (
            pd.DataFrame(index=self._base_df.index)
            .join(re_map)
            .pipe(cast_dtypes, dtype=dtype, inplace=True)[last.name]
        )
        return out

    def _fill_null(self, ser: pd.Series):
        """Full null values with dtype specific defaults."""
        if self._dtype in {str, "str"}:
            ser = ser.replace("nan", "")
        if self._dtype not in self.fill_types:
            return ser
        out = ser.fillna(self.fill_types[self._dtype])
        return out

    def dispatch(
        self,
        op,
    ):
        """
        Dispatch version operations to their appropriate functions.
        """
        # if operation is a known function
        if isinstance(op, FunctionCall):
            return self._funcs[op.name](self, *op.args, **op.kwargs)
        # if operation requests current index
        elif op in self.current_.index.names:
            return self._get_index()
        # if operation is an int or slice just get appropriate item
        elif isinstance(op, (int, slice)):  # an int gets
            return self._apply_int_index(op)
        # if a non-sliced series then follow resource_id
        elif isinstance(self.current_, pd.Series):  # assume we need to follow rid
            return self._follow_rid(op)
        # try getting child of current stored in op (attribute name)
        elif op in self.mill.schema["models"][self.cls_]:
            return self._get_children(op)
        # try returning column
        with suppress(KeyError):
            return self._get_column(op)
        # No idea, give up
        raise NotImplementedError(f"Operation Parser failed on {op}")

    def _get_index(self):
        index = self.current_.index
        self.current_ = pd.Series(index, index=index.values)

    def _apply_int_index(self, op):
        """Apply an int to aggregate list-like objects."""
        sub_struct = self._struct_df.loc[self.current_.index]
        valid_indices = get_index_group(sub_struct, op)
        self.current_ = self.current_.loc[valid_indices]
        valid_base = self.base_id_map_.isin(valid_indices)
        self.base_id_map_ = self.base_id_map_[valid_base]

    def _follow_rid(self, op):
        """current should be a series of ids, simply look them up"""
        # filter out any missing resource_ids (pandas string dtype has nulls)
        self.current_ = self.current_[~pd.isnull(self.current_)]
        out, cls = self.mill.lookup(self.current_.values)
        assert len(out) == len(self.current_)
        args = argisin(out.index.values, self.current_.values)
        # update base id map
        new = pd.Series(out.index.values, index=self.base_id_map_.index[args])
        self.base_id_map_ = new
        self.current_ = out
        self.cls_ = cls
        self.dispatch(op)

    def _get_column(self, op):
        """Try to get column from current."""
        out = self.current_[op]  # this possible exception is handled one stack up
        self.current_ = out

    def _get_children(self, op):
        """Get child objects of parent."""
        kids, cls = self.mill.get_children(self.cls_, op, df=self.current_)
        parents = self._struct_df.loc[kids.index, "parent_id"]
        # update base_id to now point to children
        args = argisin(parents.values, self.base_id_map_.values)
        new = pd.Series(kids.index.values, index=self.base_id_map_.values[args])
        self.base_id_map_ = new
        self.current_, self.cls_ = kids, cls

    def _get_dtype(self, dtype):
        """Get the datatype."""
        if get_origin(dtype) is Annotated:
            # just get raw dtype if Annotated is used
            dtype = dtype.__origin__
        return dtype

    assert set(_funcs) == (set(SUPPORTED_MODEL_OPS)), "not all funcs supported"


class _OperationSetter(_OperationResolver):
    """Class for getting a callable to set value back to mill."""
