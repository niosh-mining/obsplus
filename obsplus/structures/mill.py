"""
Module for converting tree-like structures into tables and visa-versa.
"""
from typing import Any, Type

from obsplus.structures.model import ObsPlusModel


class _DFMappingMeta(type):
    """A metaclass for df ma"""


class DFMapping:
    """
    A class for defining mappings between schema and dataframes.
    """

    _validated = False

    @classmethod
    def valdiate(cls):
        pass

    def __init_subclass__(cls, **kwargs):
        pass


class Mill:
    """
    A class for converting tree-like structures to json and back.

    Currently this just uses instances of ObsPlusModels but we plan to
    switch to awkward array in the future.
    """

    def __init__(self, data: Any, spec: Type[ObsPlusModel]):
        """

        Parameters
        ----------
        data
            Any hierarchical data which can be converted to a tree-like
            structure.
        spec
            A model defining the structure of the contained data.
        """
        self._spec = spec
        self._data = self._get_data(data)

    def _get_data(self, data):
        """Get the internal data structure."""
        return self._spec.from_orm(data)

    def __str__(self):
        msg = f"Mill with spec of {self._spec.__name__}"
        return msg


# class DFCatalogMapping:
# _base =
