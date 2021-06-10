"""
Module which provides the machinery for mapping tree-like structures
to dataframes.

May eventually replace dfextractor.
"""
import inspect
from typing import Mapping, Dict

import obsplus
from obsplus.utils.mapping import FrozenDict
from obsplus.structures.model import (
    _SpecGenerator,
    ObsPlusModel,
)
from obsplus.utils.docs import compose_docstring

from obsplus.exceptions import InvalidModelAttribute


_SUPPORTED_OPERATOR_ARGUMENTS = {
    "object": "The object represented by model",
    "mill": "The attached Mill instance",
}


class DataFramer:
    """A class for defining mappings."""

    _fields: Mapping[str, _SpecGenerator]
    _dtypes: Mapping[str, type]
    _model: ObsPlusModel
    _required_attrs = ("_model",)
    _operators = Dict[str, callable]
    _mill: "obsplus.Mill"

    def __init_subclass__(cls):
        """Validate subclasses."""
        # check required attrs
        req = {x for x in cls._required_attrs if hasattr(cls, x)}
        missing = set(cls._required_attrs) - set(req)
        if missing:
            msg = f"subclass of DataFramer missing attributes {missing}"
            raise AttributeError(msg)
        # gather up all operation trackers
        cls._fields = cls._get_fields()
        cls._dtypes = cls._get_types()
        cls._operators = cls._get_operators()

    def __init__(self, mill: "obsplus.Mill"):
        """Instantiate dataframe with a Mill."""
        self._mill = mill

    @classmethod
    def _get_fields(cls):
        """Get fields into dict and return their corresponding types."""
        fields = {}
        for i, v in vars(cls).items():
            if i.startswith("_") or not isinstance(v, _SpecGenerator):
                continue
            if not v.parent_model == cls._model:
                msg = (
                    f"attribute: {i} with op of: {v} has parent "
                    f"{v.parent_model} but Dataframer: {cls.__name__} "
                    f"requires a parent of {cls._model.__name__} "
                )
                raise InvalidModelAttribute(msg)
            fields[i] = v

        return FrozenDict(fields)

    @classmethod
    def _get_types(cls):
        """Parse the type annotations on fields, dict of such."""
        types = {i: v for i, v in cls.__annotations__.items() if i in cls._fields}
        return FrozenDict(types)

    @classmethod
    def _get_operators(cls):
        """
        Get a dict of operators supported by this dataframer.
        """
        out = {}
        for attr, value in vars(cls).items():
            if not getattr(value, "_is_obsplus_operator", False):
                continue
            out[attr] = value
        return out

    def get_dataframe(self, stash=None):
        """
        Get the dataframe defined by the dataframer.

        Parameters
        ----------
        data
            A dict of data.
        schema
            The schema describing the data structure.
        stash
            A dict of mapped components already known.

        """
        # stash = stash if stash is not None else {}
        # model_name = self._model.__name__
        # schema = self._model.get_obsplus_schema()
        # address = schema[model_name]["address"]
        # data = self._mill._data
        # # breakpoint()
        # sub_data = expand_address(data, address, stash)
        # out = []
        # df = pd.DataFrame(out)
        # return cast_dtypes(df, dtype=self._dtypes, inplace=True)


@compose_docstring(supported=_SUPPORTED_OPERATOR_ARGUMENTS)
def model_operator(method):
    """
    Register a method as an operator.

    This allows functions to be called to transverse the object tree. The
    name of the method is the string used to specify this operator.

    Supported arguments (must be named exactly as prescribed) are:
    {supported}
    """

    def _check_arguments():
        sig = inspect.signature(method)
        param_names = set(sig.parameters)
        param_names.remove("self")
        extras = param_names - set(_SUPPORTED_OPERATOR_ARGUMENTS)
        if extras:
            msg = (
                f"{extras} are not supported parameter names, supported "
                f"names are {list(_SUPPORTED_OPERATOR_ARGUMENTS)}"
            )
            raise TypeError(msg)
        assert param_names.issubset(_SUPPORTED_OPERATOR_ARGUMENTS)

    _check_arguments()
    method._is_obsplus_operator = True
    return method
