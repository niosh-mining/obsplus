"""
Module which provides the machinery for mapping tree-like structures
to dataframes.
"""


class DFMap:
    """A class for defining mappings."""

    def __init_subclass__(cls):
        """Validate subclasses."""
        base_model = getattr(cls, "_model", None)
        if base_model is None:
            msg = "subclass of DFMap must define _model attribute"
            raise AttributeError(msg)


def to_column(func):
    """Decorator for converting to columns"""


def to_tree(func):
    """Decorator for converting df to tree"""
