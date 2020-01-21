"""
Tests for general banks.
"""
from functools import lru_cache
from pathlib import Path
from typing import Union, Dict, Tuple, Optional, Mapping, Sequence

import obsplus
import obspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import pytest


bank_params = ["default_ebank", "default_wbank"]


@pytest.fixture(scope="class", params=bank_params)
def some_bank(request):
    return request.getfixturevalue(request.param)


class TestBasic:
    """
    Basic tests all banks should pass.
    """

    def test_paths(self, some_bank):
        """ Each bank should have bank paths an index paths. """
        bank_path = some_bank.bank_path
        index_path = some_bank.index_path
        assert isinstance(bank_path, Path)
        assert isinstance(index_path, Path)
