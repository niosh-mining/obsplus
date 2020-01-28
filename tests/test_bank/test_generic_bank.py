"""
Tests for general banks.
"""
from pathlib import Path

import pytest


bank_params = ["default_ebank", "default_wbank"]


@pytest.fixture(scope="class", params=bank_params)
def some_bank(request):
    """Parametrized gathering fixture for banks."""
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
