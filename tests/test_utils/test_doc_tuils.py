import textwrap

from obsplus.utils.docs import compose_docstring


class TestDocsting:
    """ tests for obsplus' simple docstring substitution function. """

    def test_docstring(self):
        """ Ensure docstrings can be composed with the docstring decorator. """
        params = textwrap.dedent(
            """
        Parameters
        ----------
        a: int
            a
        b int
            b
        """
        )

        @compose_docstring(params=params)
        def testfun1():
            """
            {params}
            """

        assert "Parameters" in testfun1.__doc__
        line = [x for x in testfun1.__doc__.split("\n") if "Parameters" in x][0]
        base_spaces = line.split("Parameters")[0]
        assert len(base_spaces) == 12
