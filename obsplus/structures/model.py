"""
Logic for defining tree-like data structures.
"""
import datetime
from functools import lru_cache
from typing import Optional, Union, Type, Sequence
from uuid import uuid4

import pandas as pd
from obspy.core import event as ev
from pydantic import root_validator, validator
from pydantic.main import BaseModel, ModelMetaclass
from pydantic.fields import SHAPE_NAME_LOOKUP
from pydantic.utils import GetterDict

from obsplus.constants import SUPPORTED_MODEL_OPS
from obsplus.exceptions import InvalidModelAttribute


class ObsPlusMeta(ModelMetaclass):
    """A mixing for the metaclass to add getitem for defining specs."""

    def __getitem__(cls: Type["ObsPlusModel"], item):
        # item_name = str(getattr(item, "__name__", item))
        cls_name = cls.__name__
        op_tuple = (cls_name, item)
        return _SpecGenerator(op_tuple, parent_model=cls)

    def __getattr__(cls: Type["ObsPlusModel"], item):
        """Attrs that don't exist can be used for making specifications."""
        fields = getattr(cls, "__fields__", {})
        # this is requesting a get_attribute operator
        if item in fields:
            return cls.__getitem__(item)
        # this is requesting a get_preferred operator
        elif item.startswith(SUPPORTED_MODEL_OPS):
            cls_name = cls.__name__
            op_tuple = (cls_name, item)
            return _SpecGenerator(op_tuple, parent_model=cls)
        msg = f"{cls.__name__} has no attribute {item}"
        raise InvalidModelAttribute(msg)


class ObsPlusModel(BaseModel, metaclass=ObsPlusMeta):
    """
    ObsPlus' base model for defining schema.
    """

    __version__ = "0.0.0"  # allows versioning of models
    __dtype__ = None  # obsplus dtype if not None
    __reference_type__ = None  # if this is a resource id, the type referred to
    # the field which contains the objects unique id
    _id_field: str = "resource_id"

    class Config:
        """pydantic config for obsplus model."""

        validate_assignment = True
        arbitrary_types_allowed = False
        orm_mode = True
        extra = "allow"

    @staticmethod
    def _convert_to_obspy(value):
        """Convert an object to obspy or return value."""
        if hasattr(value, "to_obspy"):
            return value.to_obspy()
        return value

    def to_obspy(self):
        """Convert to obspy objects."""
        name = self.__class__.__name__
        name = "ResourceIdentifier" if "ResourceIdentifier" in name else name
        cls = getattr(ev, name)
        out = {}
        # get schema and properties
        schema = self.schema()
        props = schema["properties"]
        array_props = {x for x, y in props.items() if y.get("type") == "array"}
        # iterate each property and convert back to obspy
        for prop in props:
            val = getattr(self, prop)
            if prop in array_props:
                out[prop] = [self._convert_to_obspy(x) for x in val]
            else:
                out[prop] = self._convert_to_obspy(val)
        return cls(**out)

    @classmethod
    @lru_cache()
    def get_obsplus_schema(cls) -> dict:
        """
        Return an ObsPlus Schema for this model and its children/parents.
        """
        return _get_obsplus_schema(cls)


class ResourceIdentifier(ObsPlusModel):
    """Resource ID"""

    id: Optional[str] = None
    _points_to = None
    __dtype__ = str

    @root_validator(pre=True)
    def get_id(cls, values):
        """Get the id string from the resource id"""
        if isinstance(values, GetterDict):
            value = str(values._obj)
        else:
            value = values.get("id")
        if value is None or value == "":
            value = f"smi:local/{str(uuid4())}"
        return {"id": value}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        # set the name of what this points to (if specified)
        name = cls.__name__.replace("ResourceIdentifier", "")
        cls._points_to = name or cls._points_to

    def __str__(self):
        return str(self.id)

    def __eq__(self, other):
        return str(self) == str(other)


class _ModelWithResourceID(ObsPlusModel):
    """A model which has a resource ID"""

    resource_id: Optional[ResourceIdentifier]

    @validator("resource_id", always=True)
    def get_resource_id(cls, value):
        """Ensure a valid str is returned."""
        if value is None:  # generate resource_id if empty
            return str(uuid4())
        return value


class FunctionCall:
    """Simple class for keeping track of function calls in specs."""

    def __init__(self, name, args, kwargs):
        if name not in SUPPORTED_MODEL_OPS:
            msg = f"Unsupported function {name} requested by spec."
            raise ValueError(msg)
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        msg = f"function: {self.name}, args/kwargs:{self.args}/{self.kwargs}"
        return msg

    __repr__ = __str__


class _SpecGenerator:
    """
    A class for generating specs used to access data on tree structures.

    See also :func:`obsplus.utils.mapping.resolve_tree_specs`.
    """

    def __init__(
        self,
        op_tuple: Union[Sequence[Union[int, str, FunctionCall]], "_SpecGenerator"],
        parent_model: Optional[Type[ObsPlusModel]] = None,
    ):
        if isinstance(op_tuple, self.__class__):
            self.spec_tuple = op_tuple.spec_tuple
            self.parent_model = op_tuple.parent_model
        else:
            self.spec_tuple = tuple(op_tuple)
            self.parent_model = parent_model

    def replace_first(self, new) -> tuple:
        """Replace the first operation with a new address."""
        return tuple(list(new) + list(self.spec_tuple[1:]))

    def __getattr__(self, attr):
        """Create a new instance following attributes down."""
        return self.__class__(
            list(self.spec_tuple) + [attr], parent_model=self.parent_model
        )

    def __str__(self):
        return f"Spec::{self.spec_tuple}"

    def __getitem__(self, item):
        return self.__class__(
            list(self.spec_tuple) + [item], parent_model=self.parent_model
        )

    def lookup(self):
        """Look up an ID."""
        return self

    def __call__(self, *args, **kwargs):
        fc = FunctionCall(self.spec_tuple[-1], args, kwargs)
        new_ops = tuple(list(self.spec_tuple[:-1]) + [fc])
        return self.__class__(new_ops, parent_model=self.parent_model)

    __repr__ = __str__


def _get_obsplus_schema(cls: ObsPlusModel) -> dict:
    """ "
    Return a dict used to characterize class hierarchies of models.

    Notes
    -----
    The form of the output is internal to ObsPlus and could change anytime!
    """
    DTYPES_MAP = {
        int: "Int64",
        float: "float",
        str: "str",
        datetime.datetime: "datetime64[ns]",
    }
    COL_DTYPE = {
        "model": str,
        "dtype": str,
        "referenced_model": str,
        "optional": bool,
        "is_list": bool,
    }

    def _get_dtype(type_, is_model):
        """Return pandas dtype of field"""
        raw_dtype = getattr(type_, "__dtype__", type_)
        dtype = DTYPES_MAP.get(raw_dtype, None)
        model_name = type_.__name__ if is_model else None
        return dtype, model_name

    def _is_list(shape):
        if shape not in SHAPE_NAME_LOOKUP:
            return False
        value = SHAPE_NAME_LOOKUP.get(shape, "")
        assert "List" in value
        return True

    def _is_obsplus_model(cls):
        """Return true if cls is an obsplus model subclass"""
        try:
            is_model = issubclass(cls, ObsPlusModel)
        except TypeError:
            is_model = False
        return is_model

    def _recurse(cls):
        attr_dict = {}
        for name, model_field in cls.__fields__.items():
            current = {}
            type_ = model_field.type_
            is_model = _is_obsplus_model(type_)
            current["dtype"], current["model"] = _get_dtype(type_, is_model)
            current["referenced_model"] = getattr(type_, "__reference_type__", None)
            current["optional"] = model_field.allow_none
            current["is_list"] = _is_list(model_field.shape)
            if is_model and cls.__name__ not in _tables:
                _recurse(type_)
            attr_dict[name] = current

        _tables[f"{cls.__name__}"] = attr_dict
        structure_dict = {
            "name": cls.__name__,
            "version": cls.__version__,
        }
        _meta.append(structure_dict)

    _meta = []
    _tables = {}
    _recurse(cls)
    out = {
        n: (
            pd.DataFrame(v)
            .T.astype(dtype=COL_DTYPE)
            .replace("nan", "")
            .replace("None", "")
        )
        for n, v in _tables.items()
    }
    out["__meta__"] = pd.DataFrame(_meta)
    return out
