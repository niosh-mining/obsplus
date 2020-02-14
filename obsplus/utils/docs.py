"""
Obsplus Utilities for documentation.
"""
import textwrap


def compose_docstring(**kwargs):
    """
    Decorator for composing docstrings.

    Provided values found in curly brackets in wrapped functions docstring
    will be appropriately indented and included in wrapped function's
    __docs__.
    """

    def _wrap(func):

        docstring = func.__doc__
        # iterate each provided value and look for it in the docstring
        for key, value in kwargs.items():
            search_value = "{%s}" % key
            # find all lines that match values
            lines = [x for x in docstring.split("\n") if search_value in x]
            for line in lines:
                # determine number of spaces used before matching character
                spaces = line.split(search_value)[0]
                # ensure only spaces precede search value
                assert set(spaces) == {" "}
                new = {key: textwrap.indent(textwrap.dedent(value), spaces)}
                docstring = docstring.replace(line, line.format(**new))
        func.__doc__ = docstring
        return func

    return _wrap
