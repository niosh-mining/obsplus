"""
Logic for defining tree-like data structures.
"""

from functools import lru_cache
from typing import Optional, Union, Tuple
from uuid import uuid4

from obspy.core import event as ev
from pydantic import root_validator, validator
from pydantic.main import BaseModel, ModelMetaclass

from obsplus.exceptions import InvalidModelOperation, InvalidModelAttribute


# a global cache for graphs

#
class ObsPlusMeta(ModelMetaclass):
    """A mixing for the metaclass to add getitem."""

    # def __getitem__(cls, item):
    #     item_name = str(getattr(item, '__name__', item))
    #     cls_name = cls.__name__
    #     name = f"{cls_name}__{item_name}"
    #
    #     _dict = {
    #         '__reduce__': lambda cls: f"{cls.__module__}.{cls.__name__}"
    #         # '__getstate__': ModelMetaclass.__getstate__,
    #         # "__setstate__": ModelMetaclass.__setstate__,
    #     }
    #     new = type(name, (cls,), _dict)
    #
    #
    #     return new
    #
    # def __getstate__(cls):
    #     breakpoint()
    #
    # def __setstate__(cls):
    #     breakpoint()
    def __getattr__(cls: "ObsPlusModel", item):
        if item in getattr(cls, "__fields__", {}):
            cls_name = cls.__name__
            op_str = f"{cls_name}.{item}"
            return OperationTracker(op_str)

        msg = f"{cls.__name__} has not attribute {item}"
        raise AttributeError(msg)


class ObsPlusModel(BaseModel, metaclass=ObsPlusMeta):
    # extra: Optional[Dict[str, Any]] = None
    _contains = None  # for storing the containing info.

    class Config:
        pass
        validate_assignment = True
        arbitrary_types_allowed = True
        orm_mode = True
        extra = "allow"

    @staticmethod
    def _convert_to_obspy(value):
        """Convert an object to obspy or return value. """
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
    def get_graph_dict(cls) -> dict:
        """return a mapper for this model."""
        return _get_graph_dict(cls.schema())


class ResourceIdentifier(ObsPlusModel):
    """Resource ID"""

    id: Optional[str] = None
    _points_to = None

    @root_validator(pre=True)
    def get_id(cls, values):
        """Get the id string from the resource id"""
        value = values.get("id")
        if value is None:
            value = str(uuid4())
        return {"id": value}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        # set the name of what this points to (if specified)
        name = cls.__name__.replace("ResourceIdentifier", "")
        cls._points_to = name or cls._points_to


class _ModelWithResourceID(ObsPlusModel):
    """A model which has a resource ID"""

    resource_id: Optional[ResourceIdentifier]

    @validator("resource_id", always=True)
    def get_resource_id(cls, value):
        """Ensure a valid str is returned. """
        if value is None:
            return str(uuid4())
        return value


class OperationTracker:
    """
    A simple class for keeping track of operations on a model.
    """

    def __init__(self, op_str: Union[str, "OperationTracker"]):
        if isinstance(op_str, self.__class__):
            self._op_str = op_str._op_str
        self._op_str = op_str
        self._validated = False
        self._type, self._is_array = None, None

    def __getattr__(self, attr):
        """Create a new instance following attributes down."""
        return self.__class__(f"{self._op_str}.{attr}")

    def __str__(self):
        return f"AttributeTracker:: {self._op_str}"

    def __or__(self, other):
        if " | " in self._op_str:
            msg = "Only one '|' operation is currently permitted."
            raise InvalidModelOperation(msg)
        new_str = f"({self._op_str} | {other._op_str})"
        return self.__class__(new_str)

    def __getitem__(self, item):
        if not isinstance(item, int):
            msg = (
                "getitem is reserved for selecting elements from as list "
                f"and therefore must be an int. You passed {item}"
            )
            raise TypeError(msg)
        return self.__class__(f"{self._op_str}[{item}]")

    __repr__ = __str__

    def validate(self, graph_dict):
        """
        Validate the operation tracker based on a graph dict.

        Parameters
        ----------
        graph_dict
            The dict defining tree structure returned from
            ~:meth:ObsPlusModel.get_graph_dict.

        Raises
        ------
        InvalidModelOperation if the requested operations are inconsistent
        with the object model in graph_dict.
        """

        # first resolve | by slicing out both strings
        op_str = self._op_str
        if " | " in op_str:
            paren_index1 = op_str.index("(")
            paren_index2 = op_str.index(")")
            inside_or_str = op_str[paren_index1 + 1 : paren_index2]
            resulting_type = self._validate_or(inside_or_str, graph_dict)
            op_str = op_str.replace(f"({inside_or_str})", resulting_type)

        # check the attributes and type.
        self._type, self._is_array = _get_type_from_graph(op_str, graph_dict)
        self._validated = True
        return self

    def _validate_or(self, or_str, graph_dict):
        """"""
        str_1, str_2 = or_str.split(" | ")
        # first make sure attributes exist and are valid.
        obj1 = OperationTracker(str_1).validate(graph_dict)
        obj2 = OperationTracker(str_2).validate(graph_dict)
        # then make sure types are corrct.
        type1 = obj1._get_type(graph_dict)
        type2 = obj2._get_type(graph_dict)
        if type1 != type2:
            msg = (
                f"Invalid | found in {or_str}. {obj1} is of type {type1} "
                f"and {obj2} is of type {type2}, they must be the same."
            )
            raise InvalidModelOperation(msg)
        return type1[0]

    def _get_type(self, graph_dict) -> Tuple[str, bool]:
        """Get a string of the type. """
        # first split on attrs
        assert " | " not in self._op_str, "or is not supported at this stage."
        return _get_type_from_graph(self._op_str, graph_dict)


def _get_type_from_graph(address_string: str, graph_dict: dict) -> Tuple[str, bool]:
    """
    Get the type an address string refers to from a graph_dict.

    Parameters
    ----------
    address_string
        A string pointing to an object defined in graph.
    graph_dict
        A dict of the object heirarch returned by ~:meth:ObsPlusModel.get_graph_dict.

    Raises
    ------
    InvalidModelOperation if the path is invalid for the schema.

    Returns
    -------
    A tuple of ('type': str, 'is_array': bool)
    """

    def _get_type_from_entry(attr_dict, attr_name):
        """Get the type from an attr_dict """
        atype = attr_dict["attr_type"][attr]
        ref = attr_dict["attr_ref"].get(attr)
        ref_id = attr_dict["attr_id_ref"].get(attr)
        is_array = attr_dict["attr_is_array"].get(attr, False)
        if ref:
            atype = ref
        if ref_id:
            atype = ref_id
        return atype, is_array

    # split attributes, determine which refer to arrays
    attr_list = address_string.split(".")
    has_getitem = ["[" in x and "]" in x for x in attr_list[1:]]
    # init values and iterate through each level of attribute map
    current_parent_type = atype = attr_list[0]
    array = False
    for attr, has_getitem in zip(attr_list[1:], has_getitem):
        if has_getitem:
            attr = attr[: attr.index("[")]  # strip out []
        current = graph_dict[current_parent_type]
        try:
            atype, array = _get_type_from_entry(current, attr)
        except KeyError:
            msg = f"{current_parent_type} has no attribute {attr}"
            raise InvalidModelAttribute(msg)
        current_parent_type = atype
        # determine if an array is being reduced
        if array and has_getitem:
            array = False
    return atype, array


def _get_attr_ref_type(attr_dict):
    """
    Determine the type and reference type of an attribute from a schema dict.

    Returns a tuple of (
    """
    ref_str = "#/definitions/"
    atype = attr_dict.get("type", "object")
    is_array = atype == "array"
    ref = None
    if is_array:  # dealing with array, find out sub-type
        items = attr_dict.get("items", {})
        ref = items.get("$ref", None)
        atype = items.get("type", atype)
    else:
        ref = attr_dict.get("$ref", None)
    if ref:  # there is a reference to a def, just get name
        ref = ref.replace(ref_str, "")
    # check if this is a datetime
    if attr_dict.get("format", "") == "date-time":
        atype = "datetime64[ns]"
    return atype, ref, is_array


def _get_graph_dict(schema: ObsPlusModel) -> dict:
    """"
    Return a dict of the form:

    #TODO fill in form of dict
    """
    # type_dict = {'string': str, 'array': list, 'float': float, 'int': int}
    rid_str = "ResourceIdentifier"

    def _init_empty():
        """Return an empty form of dict."""
        return {
            "address": [],
            "attr_type": {},
            "attr_is_array": {},
            "attr_ref": {},
            "attr_id_ref": {},
        }

    def _recurse(base, definitions, cls_dict=None, address=()):
        # init empty data structures or reuse
        cls_dict = cls_dict if cls_dict is not None else {}
        title = base.get("title", "")
        # get the class dictionary file
        current = _init_empty() if title not in cls_dict else cls_dict[title]
        # add address and return if attributes have already been parsed
        current["address"].append(tuple(address))
        if current["attr_type"]:
            return
        # next iterate attributes
        for attr, attr_dict in base["properties"].items():
            # this attribute has no linked items
            atype, ref, is_array = _get_attr_ref_type(attr_dict)
            current["attr_type"][attr] = atype
            current["attr_is_array"][attr] = is_array
            # this is not object which should have a definition.
            if ref is None:
                continue
            # add reference type
            current["attr_ref"][attr] = ref
            # add name resource_id points to, if resource_id
            cadd = [attr] if atype == "array" else attr
            new_address = list(address) + [cadd]
            new_base = definitions[ref]
            _recurse(new_base, definitions, cls_dict, new_address)
            # add resource_id info
            if rid_str in ref:
                name = ref.replace(rid_str, "")
                if name in definitions:
                    current["attr_id_ref"][attr] = name
        cls_dict[title] = current
        return cls_dict

    return _recurse(schema, schema["definitions"])
