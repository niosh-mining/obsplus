"""
Module which provides the machinery for mapping tree-like structures
to dataframes.
"""

from pathlib import Path
from functools import lru_cache
from typing import Union, Optional, List, Mapping, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
import obsplus
import obspy
import pandas as pd
import pytest


class DFMap:
    """A class for defining mappings."""

    def __init_subclass__(cls):
        """Validate subclasses."""
        base_model = getattr(cls, "_model", None)
        if base_model is None:
            msg = f"subclass of DFMap must define _model attribute"
            raise AttributeError(msg)


def to_column(func):
    """Decorator for converting to columns"""


def to_tree(func):
    """Decorator for converting df to tree"""
