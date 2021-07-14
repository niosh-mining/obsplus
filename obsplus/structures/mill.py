"""
Module for converting tree-like structures into tables and visa-versa.
"""
import copy
from typing import Set
from typing_extensions import Annotated, get_origin
import uuid
from collections import defaultdict
from contextlib import suppress
from functools import lru_cache, wraps
from typing import Type, Dict, Optional, Tuple, Sequence, TypeVar

import numpy as np
import pandas as pd

import obsplus
from obsplus.constants import SUPPORTED_MODEL_OPS
from obsplus.exceptions import IncompatibleDataFramesError, InvalidModelAttribute
from obsplus.structures.model import ObsPlusModel, FunctionCall
from obsplus.utils.pd import (
    cast_dtypes,
    order_columns,
    get_index_group,
    int64_to_int_obj,
)
from obsplus.utils.misc import argisin, register_func

MillType = TypeVar("MillType", bound="Mill")


def _inplace_or_copy(func):
    """Decorator for performing a function inplace or copying self."""

    @wraps(func)
    def _func(self, *args, **kwargs):
        inplace = kwargs.get("inplace", False)
        if inplace:
            kwargs["inplace"] = False
            new = self.copy()
            new_func = getattr(new, func.__name__)
            return new_func(*args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return _func


class Mill:
    """
    A class for managing tree-like data structures with table slices.

    Currently this just uses instances of ObsPlusModels but we plan to
    switch to awkward array in the future.
    """

    _model: Type[ObsPlusModel] = ObsPlusModel
    _id_name: Optional[str] = None
    _data: dict
    _dataframers: Dict[str, Type["obsplus.DataFramer"]]
    _structure_key: str = "__structure__"
    _df_dicts = None
    _scope_model_name = None

    def __init_subclass__(cls, **kwargs):
        """Ensure subclasses have their own framers dict."""
        cls._dataframers = {}

    def __init__(self, data):
        self._data = data

    @classmethod
    def _from_df_dict(cls, df_dict) -> MillType:
        """init instance from df dict."""
        out = cls(None)
        out._df_dicts = df_dict
        return out

    @property
    @lru_cache()
    def _schema(self):
        """Get schema from model."""
        return _get_schema_dicts(self._schema_df)

    @property
    @lru_cache()
    def _schema_df(self):
        """Get schema from model."""
        return self._model.get_obsplus_schema()

    def _get_table_dict(self):
        if self._df_dicts is not None:
            return self._df_dicts
        df_dicts = _dict_to_tables(
            data=self._get_data(self._data),
            schema=self._schema,
            data_type=self._model.__name__,
            scope_type=self._scope_model_name,
        )
        df_dicts = self._post_df_dict(df_dicts)
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
        if name in self._get_table_dict():
            return self._get_table_dict()[name]
        try:
            Framer = self._dataframers[name]
        except KeyError:
            msg = (
                f"Unknown dataframe: {name}, known dataframes are: \n"
                f"{list(self._dataframers) + list(self._get_table_dict())}"
            )
            raise KeyError(msg)

        framer = Framer()
        return self._get_df_from_framer(framer)

    def get_summary_df(self):
        """Return a dataframe which summarizes objects on defined level."""
        return self.get_df("__summary__")

    def _get_df_from_framer(self, framer):
        out = {}
        base = framer._model_name
        base_df = self.get_df(base)
        for attr_name, tracker in framer._fields.items():
            specs = tracker.spec_tuple
            dtype = framer._dtypes.get(attr_name)
            resolver = _OperationResolver(self, specs, base_df, base, dtype)
            out[attr_name] = resolver()
        df = pd.DataFrame(out)
        return df

    def __str__(self):
        cls_name = self.__class__.__name__
        model_name = self._model.__name__
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
            return pd.DataFrame(), ""
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
            model_name = self._schema["models"][cls][attr_name]
            child_table = self.get_df(model_name)
        except KeyError:
            msg = f"{cls} has no model attributes of {attr_name}"
            raise InvalidModelAttribute(msg)
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

    def _get_structure_df(self, scope_ids=None):
        """
        Get the df defining object structure.

        Optionally, filter structure based on scope_ids.
        """
        df = self._df_dicts["__structure__"]
        if scope_ids is not None:
            df = df[df["scope_id"].isin(scope_ids)]
        return df

    def _post_df_dict(self, df_dicts):
        """This can be subclassed to run validators/checks on dataframes."""
        return df_dicts

    def to_model(self, scope_ids=None):
        """Return a populated model with data from mill."""
        schema = self._schema
        data = _tables_to_dict(
            table_dict=self._get_table_dict(),
            cls_name=self._model.__name__,
            schema=schema,
            structure_df=self._get_structure_df(scope_ids=scope_ids),
        )
        return self._model(**data)

    def get_scope_ids(self, **kwargs) -> Optional[Set[str]]:
        """
        Get scope ids given a query.

        This only works if the summary dataframe is defined.
        """

    def copy(self, deep=False):
        """Return a copy of the mill."""
        return self.__class__._from_df_dict(copy.deepcopy(self._get_table_dict()))


def _get_schema_dicts(df_schema):
    """Parses out dicts from dataframes for use in decomposition"""
    rid_dict, model_dict, dtype_dict, array_dict = {}, {}, {}, {}
    for name, df in df_schema.items():
        if name.startswith("__"):
            continue
        new_name = name.replace("schema_", "")
        # get set of attrs which are resource ids
        is_rid = df["model"].str.startswith("ResourceIdentifier")
        rid_dict[new_name] = set(df[is_rid].index)
        # get referenced model type
        has_model = df["model"].astype(bool)
        model_dict[new_name] = dict(df[has_model]["model"])
        # dtype
        has_dtype = df["dtype"].astype(bool)
        dtype_dict[new_name] = dict(df[has_dtype]["dtype"])
        array_dict[new_name] = set(df.index[df["is_list"]])
    out = {
        "resource_ids": rid_dict,
        "models": model_dict,
        "dtypes": dtype_dict,
        "arrays": array_dict,
    }
    return out


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

    def _make_df_dict(object_list):
        """Convert a dist of lists of dicts into a dict of DFs."""
        out = {}
        for class_name, data in object_list.items():
            if class_name.startswith("ResourceIdentifier"):
                continue
            dtypes = dtype_dict[class_name]
            dff = (
                pd.DataFrame(data)
                .pipe(order_columns, sorted(dtypes))
                .pipe(cast_dtypes, dtypes)
                .replace({"nan": ""})  # TODO could make this a bit faster
                .set_index(id_field)
            )
            out[class_name] = dff

        out["__structure__"] = pd.DataFrame(structure_list).set_index(id_field)
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
            if name.startswith(("__", "schema")):
                continue
            # sort rows so that objects are reassembled correctly
            sort_cols = ["parent_id", "attr", "index"]
            inds = np.intersect1d(struct.index, df.index, assume_unique=True)
            sub_struct = struct.loc[inds].sort_values(sort_cols)
            df = df.loc[sub_struct.index]
            # get values that should be None
            con2 = (df.isnull() | (df == "").fillna(False)).astype(bool)
            sub = (
                df.pipe(int64_to_int_obj)
                .astype(object)
                .where(~con2, None)
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


class _OperationResolver:
    """
    A class for resolving operations specifying data access from models.
    """

    fill_types = {
        str: "",
        "str": "",
    }
    _funcs = {}

    def __init__(self, mill: Mill, spec, base_df, base_cls_name, dtype):
        self.mill = mill
        self.spec = spec if isinstance(spec, tuple) else spec.spec_tuple
        self._dtype = self._get_dtype(dtype)
        self._struc_df = mill.get_df("__structure__")
        # get base (starting place) info
        self._base_df = base_df
        self._base_cls = base_cls_name
        # set attrs which update for each operation
        self.current_ = base_df
        self.cls_ = base_cls_name
        # maps the id of current back to base objects
        self.base_id_map_ = pd.Series(base_df.index, index=base_df.index)

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
        parents_id_map = self._struc_df.loc[self.current_.index, "parent_id"]
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
        struct = self._struc_df.loc[df.index]
        last_ids = struct.groupby(["parent_id", "attr"])["index"].idxmax()
        self._reduce_current(df.loc[last_ids.values])

    @register_func(_funcs, "first")
    def _get_first(self):
        """return the first of each item (by index, attr, parent_id)."""
        df = self.current_
        struct = self._struc_df.loc[df.index]
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

    def __call__(self):
        """Resolve operations on mill."""
        # apply all operation
        for op in self.spec[1:]:
            self.dispatch(op)
            # ensure base_id map stays updated
            assert set(self.base_id_map_.values) == set(self.current_.index.values)
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
            .pipe(cast_dtypes, dtype=dtype)[last.name]
            .pipe(self._fill_null)
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
        # try returning column
        with suppress(KeyError):
            return self._get_column(op)
        # try getting child of current stored in op (attribute name)
        with suppress(InvalidModelAttribute):
            return self._get_children(op)
        # No idea, give up
        raise NotImplementedError("How did you get here!?")

    def _get_index(self):
        index = self.current_.index
        self.current_ = pd.Series(index, index=index.values)

    def _apply_int_index(self, op):
        """Apply an int to aggregate list-like objects."""
        sub_struct = self._struc_df.loc[self.current_.index]
        valid_indices = get_index_group(sub_struct, op)
        self.current_ = self.current_.loc[valid_indices]
        valid_base = self.base_id_map_.isin(valid_indices)
        self.base_id_map_ = self.base_id_map_[valid_base]

    def _follow_rid(self, op):
        """current should be a series of ids, simply look them up"""
        # filter out any missing resource_ids
        self.current_ = self.current_[self.current_.astype(bool)]
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
        parents = self._struc_df.loc[kids.index, "parent_id"]
        # update base_id to now point to children
        args = argisin(parents.values, self.base_id_map_.values)
        new = pd.Series(kids.index.values, index=self.base_id_map_.values[args])
        self.base_id_map_ = new
        self.current_, self.cls_ = kids, cls

    def _get_dtype(self, dtype):
        """Get the datatype."""
        if get_origin(dtype) is Annotated:
            # just get raw dtype
            dtype = dtype.__origin__
        return dtype

    assert set(_funcs) == (set(SUPPORTED_MODEL_OPS)), "not all funcs supported"


def _invert_series(ser):
    """Invert a series, return new series."""
    return pd.Series(ser.index.values, index=ser.values)
