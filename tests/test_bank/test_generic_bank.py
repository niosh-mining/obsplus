"""
Tests for general banks.
"""
from pathlib import Path

import numpy as np
import obspy
import pytest

import obsplus
from obsplus.utils.misc import suppress_warnings

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


class TestVersions:
    """Tests related to versioning."""

    high_version_str: str = "1000.0.0"

    @pytest.fixture
    def ebank_high_version(self, tmpdir, monkeypatch):
        """ return the default bank with a negative version number. """
        # monkey patch obsplus version so that a low version is saved to disk
        monkeypatch.setattr(obsplus, "__last_version__", self.high_version_str)
        cat = obspy.read_events()
        ebank = obsplus.EventBank(tmpdir).put_events(cat, update_index=False)
        # write index
        with suppress_warnings():
            ebank.update_index()
        monkeypatch.undo()
        assert ebank._index_version == self.high_version_str
        assert obsplus.__last_version__ != self.high_version_str
        return ebank

    def test_future_version(self, ebank_high_version):
        """Ensure reading a bank with a future version issues warning."""
        path = ebank_high_version.bank_path
        with pytest.warns(UserWarning) as w:
            obsplus.EventBank(path)
        assert len(w) == 1
        message = w.list[0].message.args[0]
        assert "a newer version of ObsPlus" in message
