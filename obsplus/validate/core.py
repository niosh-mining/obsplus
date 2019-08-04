"""
Core modules for validate.
"""
from contextlib import contextmanager
from functools import singledispatch
from typing import Optional

import pandas as pd


@contextmanager()
def temp_validate_namespace():
    """
    A contextmanager for setting temporary namespaces.
    """
    yield


@singledispatch
def decompose(obj):
    """ Decompose an object into its constitutive parts. """


def validate(obj, namespace) -> Optional[pd.DataFrame]:
    pass


def validator(namespace: str, cls: type):
    """
    Register a callabl to a given namespace to operate on type cls.

    Parameters
    ----------
    namespace

    cls

    Returns
    -------

    """
