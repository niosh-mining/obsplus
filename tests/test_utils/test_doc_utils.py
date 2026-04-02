"""Tests for docstring utils."""

import textwrap

import pytest
from obsplus.constants import STATION_DTYPES
from obsplus.exceptions import DocstringCompositionError
from obsplus.utils.docs import compose_docstring, format_dtypes


class TestDocsting:
    """tests for obsplus' simple docstring substitution function."""

    def count_white_space(self, some_str):
        """Count the number of whitespace chars in a str"""
        return len(some_str) - len(some_str.lstrip(" "))

    def test_docstring(self):
        """Ensure docstrings can be composed with the docstring decorator."""
        params = textwrap.dedent(
            """
        Parameters
        ----------
        a
            a
        b
            b
        """
        )

        @compose_docstring(params=params)
        def testfun1():
            """
            A simple test fucntion.

            {params}
            """

        assert "Parameters" in testfun1.__doc__
        line = next(x for x in testfun1.__doc__.split("\n") if "Parameters" in x)
        base_spaces = line.split("Parameters")[0]
        # py3.13+ automatically strips white space from docstrings so 12
        # and 0 are valid lengths.
        assert len(base_spaces) in {12, 0}

    def test_list_indent(self):
        """Ensure lists are indented equally."""
        str_list = ["Hey", "who", "moved", "my", "cheese!?"]

        @compose_docstring(params=str_list)
        def dummy_func():
            """
            Some useful information indeed:
                {params}
            """

        doc_str_list = dummy_func.__doc__.split("\n")
        # the number of spaces between each list element should be the same.
        list_lines = doc_str_list[2:-1]
        white_space_counts = [self.count_white_space(x) for x in list_lines]
        # all whitespace counts should be the same for the list lines.
        assert len(set(white_space_counts)) == 1

    def test_raises_on_unused_key(self):
        """Unused compose_docstring keys should fail fast."""
        with pytest.raises(DocstringCompositionError, match="unused keys"):

            @compose_docstring(params="value")
            def testfun2():
                """
                A simple test function.
                """

    def test_raises_on_unresolved_placeholder(self):
        """Unresolved identifier placeholders should raise."""
        with pytest.raises(DocstringCompositionError, match="unresolved placeholders"):

            @compose_docstring(params="value")
            def testfun3():
                """
                A simple test function.

                {params}
                {missing_value}
                """

    def test_ignores_literal_non_placeholder_braces(self):
        """Literal braces that are not placeholders should not raise."""

        @compose_docstring(params="value")
        def testfun4():
            """
            Example dictionary:
            {"a": 1}

            {params}
            """

        assert "value" in testfun4.__doc__

    def test_exception_names_function_and_keys(self):
        """The error should identify the function and bad keys."""
        with pytest.raises(DocstringCompositionError) as exc:

            @compose_docstring(params="value")
            def testfun5():
                """
                A simple test function.

                {missing_value}
                """

        msg = str(exc.value)
        assert "testfun5" in msg
        assert "params" in msg
        assert "missing_value" in msg


class TestFormatDtypes:
    """Tests for formatting datatypes to display in docstrings."""

    def test_formatting(self):
        """Test for formatting StationDtypes."""
        out = format_dtypes(STATION_DTYPES)
        assert "\nstation: nslc_code" in out
