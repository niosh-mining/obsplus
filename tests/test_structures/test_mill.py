"""
A module for testing the mill.
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


from obsplus.events.schema import Catalog as CatalogSchema
from obsplus.structures.mill import Mill


@pytest.fixture(scope='class')
def event_mill():
    """Init a mill from an event."""
    cat = obsplus.load_dataset('bingham_test').event_client.get_events()
    mill = Mill(cat, CatalogSchema)
    return mill


class TestDFMapper:
    """tests for mapping dataframes to tree structures."""


class TestMillBasics:
    """Tests for the basics of the mill."""

    def test_str(self, event_mill):
        """Ensure a sensible str rep is available."""
        breakpoint()
        str_rep = str(event_mill)
