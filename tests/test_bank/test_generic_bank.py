"""
Tests for general banks.
"""
from pathlib import Path

import numpy as np
import obspy
import pytest

import obsplus


bank_params = ["default_ebank", "default_wbank"]


@pytest.fixture(scope="class", params=bank_params)
def some_bank(request):
    """Parametrized gathering fixture for banks."""
    return request.getfixturevalue(request.param)


class TestBasic:
    """Basic tests all banks should pass."""

    def test_paths(self, some_bank):
        """ Each bank should have bank paths and index paths. """
        bank_path = some_bank.bank_path
        index_path = some_bank.index_path
        assert isinstance(bank_path, Path)
        assert isinstance(index_path, Path)

    def test_load_example_event_bank(self, tmp_path):
        """Test for loading the example banks."""
        ebank = obsplus.EventBank.load_example_bank(path=tmp_path)
        assert isinstance(ebank, obsplus.EventBank)
        cat = ebank.get_events()
        assert isinstance(cat, obspy.Catalog)
        assert len(cat)

    def test_load_example_wave_bank(self, tmp_path):
        """Test for loading the example banks."""
        wbank = obsplus.WaveBank.load_example_bank(path=tmp_path)
        assert isinstance(wbank, obsplus.WaveBank)
        st = wbank.get_waveforms()
        assert isinstance(st, obspy.Stream)
        assert len(st)

    def test_repr(self, default_ebank):
        """Tests for the repr method."""
        out = repr(default_ebank)
        assert "EventBank" in out

    def test_last_updated(self, default_ebank):
        """Last updated should be a datetime64"""
        last_update = default_ebank.last_updated
        assert isinstance(last_update, np.datetime64)
