"""
Ensure the interface isinstance and issubclass methods work
"""
import obspy
import pytest
from obspy.clients.fdsn.client import Client, FDSNException

from obsplus import EventBank, WaveBank
from obsplus.utils.misc import _get_progressbar
from obsplus.interfaces import (
    EventClient,
    WaveformClient,
    StationClient,
    ProgressBar,
)


class DynamicWrapper:
    """Simple wrapper for an object."""

    def __init__(self, obj):
        self.obj = obj

    def __getattr__(self, item):
        return getattr(self.obj, item)


# fixtures


@pytest.fixture(scope="session")
def iris_client():
    """return the IRIS client"""
    try:
        return Client()
    except FDSNException:
        pytest.skip("could not connect to webservice")


class TestEventClient:
    """Tests for event client interface."""

    not_event_client_instances = ["a", 1]

    def test_fdsn_isinstance(self, iris_client):
        """ensure the client is an instance of EventClient"""
        assert isinstance(iris_client, EventClient)

    def test_fdsn_issubclass(self):
        """Test client FDSN client meets EventClient interface."""
        issubclass(str, EventClient)
        assert issubclass(Client, EventClient)

    def test_catalog(self):
        """ensure a events is also an EventClient"""
        cat = obspy.read_events()
        assert isinstance(cat, EventClient)
        assert issubclass(obspy.Catalog, EventClient)

    def test_eventbank(self, bingham_dataset):
        """Ensure eventbank meets event client interface."""
        ebank = EventBank(bingham_dataset.event_path)
        assert isinstance(ebank, EventClient)
        assert issubclass(EventBank, EventClient)

    def test_dynamic_client(self, bingham_dataset):
        """Ensure dynamic instances work."""
        event_client = DynamicWrapper(bingham_dataset.event_client)
        assert isinstance(event_client, EventClient)

    @pytest.mark.parametrize("not_client", not_event_client_instances)
    def test_not_instances(self, not_client):
        """Ensure a few negative examples work."""
        assert not isinstance(not_client, EventClient)


class nTestWaveformClient:
    """Tests for waveform client interface."""

    def test_fdsn_isinstance(self, iris_client):
        """ensure the client is an instance of EventClient"""
        assert isinstance(iris_client, WaveformClient)
        assert not isinstance(10, WaveformClient)

    def test_fdsn_issubclass(self):
        """Ensure fdsn client meets interface."""
        assert issubclass(Client, WaveformClient)
        assert not issubclass(str, WaveformClient)

    def test_stream(self):
        """Ensure Stream meet client interface."""
        st = obspy.read()
        assert isinstance(st, WaveformClient)
        assert issubclass(obspy.Stream, WaveformClient)

    def test_wavebank(self, bingham_dataset):
        """Ensure wavebank meets waveform client interface."""
        wavebank = bingham_dataset.waveform_client
        assert isinstance(wavebank, WaveformClient)
        assert issubclass(WaveBank, WaveformClient)

    def test_dynamic_client(self, bingham_dataset):
        """Ensure dynamic instances work."""
        waveform_client = DynamicWrapper(bingham_dataset.waveform_client)
        assert isinstance(waveform_client, WaveformClient)


class TestStationClient:
    """Tests for station client interface."""

    def test_fdsn_isinstance(self, iris_client):
        """ensure the client is an instance of EventClient"""
        assert isinstance(iris_client, StationClient)
        assert not isinstance(10, StationClient)

    def test_fdsn_issubclass(self):
        """FDSN client should be a sublcass of station client."""
        assert issubclass(Client, StationClient)
        assert not issubclass(str, StationClient)

    def test_inventory(self):
        """Tests inventory meet station client interface."""
        inv = obspy.read_inventory()
        assert isinstance(inv, StationClient)
        assert issubclass(obspy.Inventory, StationClient)

    def test_dynamic_client(self, bingham_dataset):
        """Ensure dynamic instances work."""
        station_client = DynamicWrapper(bingham_dataset.station_client)
        assert isinstance(station_client, StationClient)


class TestBar:
    """Tests the progressbar interface."""

    def test_progressbar_isinstance(self):
        """Ensure the ProgressBar2 ProgressBar is an instance."""
        ProgBar = _get_progressbar()
        assert issubclass(ProgBar, ProgressBar)

    def test_custom_progress_bar(self):
        """Ensure custom progress bar works as well."""

        class MyBar:
            def update(self, num):
                pass

            def finish(self):
                pass

        assert issubclass(MyBar, ProgressBar)
        assert isinstance(MyBar(), ProgressBar)

    def test_malformed_progress_bar(self):
        """
        Ensure a ProgressBar implementation missing methods is not subclass.
        """

        class MyBadBar:
            def update(self, num):
                pass

        assert not issubclass(MyBadBar, ProgressBar)
        assert not isinstance(MyBadBar(), ProgressBar)
