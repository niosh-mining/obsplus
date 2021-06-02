"""
Module for converting tree-like structures into tables and visa-versa.
"""
from typing import Type


from obsplus.structures.model import ObsPlusModel


class _DFMappingMeta(type):
    """A metaclass for df ma"""


class DataFramer:
    """
    A class for defining conversions between trees and tables.
    """

    _base: Type[ObsPlusModel]
    _validated = False

    def __init_subclass__(cls, **kwargs):
        pass


class Mill:
    """
    A class for managing tree-like data structures.

    Currently this just uses instances of ObsPlusModels but we plan to
    switch to awkward array in the future.
    """

    _spec: Type[ObsPlusModel] = ObsPlusModel
    _id_field = "resource_id"
    _data: ObsPlusModel
    _dataframers = {}

    def _get_data(self, data):
        """Get the internal data structure."""
        return self._spec.from_orm(data)

    def __str__(self):
        msg = f"Mill with spec of {self._spec.__name__}"
        return msg

    @classmethod
    def register_data_framer(cls, name):
        """
        Register a dataframer on this mill.
        """

        def _func(framer: DataFramer):
            cls._dataframers[name] = framer
            return framer

        # TODO add check for already defined mappers
        return _func


#
# class EventMill:
#     """
#     An alternative datastructure for working with seismic events.
#
#     Maintains compatibility with ObsPy's QML-based `Catalog`.
#     """
#     _spec =


# class DFCatalogMapping:
# _base =
