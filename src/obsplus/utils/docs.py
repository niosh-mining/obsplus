"""
Obsplus Utilities for documentation.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from typing import Any


def compose_docstring(**kwargs: str | Sequence[str]):
    """
    Decorator for composing docstrings.

    This allows components of docstrings which are often repeated to be
    specified in a single place. Values provided to this function should
    have string keys and string or list values. Keys are found in curly
    brackets in the wrapped functions docstring and their values are
    substituted with proper indentation.

    Notes
    -----
    A function's docstring can be accessed via the `__docs__` attribute.

    Examples
    --------
    @compose_docstring(some_value='10')
    def example_function():
        '''
        Some useful description

        The following line will be the string '10':
        {some_value}
        '''
    """

    def _wrap(func):
        docstring = func.__doc__
        # iterate each provided value and look for it in the docstring
        for key, value in kwargs.items():
            value = value if isinstance(value, str) else "\n".join(value)
            # strip out first line if needed
            value = value.lstrip()
            search_value = f"{{{key}}}"
            # find all lines that match values
            lines = [x for x in docstring.split("\n") if search_value in x]
            for line in lines:
                # determine number of spaces used before matching character
                spaces = line.split(search_value)[0]
                # ensure only spaces precede search value
                assert set(spaces) == {" "}
                new = textwrap.indent(textwrap.dedent(value), spaces)
                docstring = docstring.replace(line, new)

        func.__doc__ = docstring
        return func

    return _wrap


def format_dtypes(dtype_dict: dict[str, Any]) -> str:
    """
    Convert a dictionary of datatypes to a string printable format for
    displaying in docstrings.

    Parameters
    ----------
    dtype_dict
        A dict of columns (keys) and values (dtypes).

    Returns
    -------
    A string formatted for :func:`obsplus.utils.docs.compose_docstring`.
    """

    def _format_cls_str(cls):
        """Format str representation of classes to be nicer on the eyes."""
        cls_str = str(cls)
        return cls_str.replace("<class ", "").replace(">", "").replace("'", "")

    str_list = [f"{x}: {_format_cls_str(y)}" for x, y in dtype_dict.items()]
    out = "\n".join(str_list)
    return out
