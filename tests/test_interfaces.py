"""
Ensure the interface isinstance and issubclass methods work
"""
import obspy
import pytest

from obsplus import EventBank, WaveBank
from obsplus.interfaces import EventClient, WaveformClient, StationClient
from obspy.clients.fdsn import Client


# fixtures


@pytest.fixture(scope="session")
def iris_client():
    """ return the IRIS client """
    return Client()


class TestEventClient:
    def test_fdsn_isinstance(self, iris_client):
        """ ensure the client is an instance of EventClient """
        assert isinstance(iris_client, EventClient)
        assert not isinstance(10, EventClient)

    def test_fdsn_issubclass(self):
        assert issubclass(Client, EventClient)
        assert not issubclass(str, EventClient)

    def test_catalog(self):
        """ ensure a events is also an EventClient """
        cat = obspy.read_events()
        assert isinstance(cat, EventClient)
        assert issubclass(obspy.Catalog, EventClient)

    def test_eventbank(self, bingham_dataset):
        ebank = EventBank(bingham_dataset.event_path)
        assert isinstance(ebank, EventClient)
        assert issubclass(EventBank, EventClient)


class TestWaveformClient:
    def test_fdsn_isinstance(self, iris_client):
        """ ensure the client is an instance of EventClient """
        assert isinstance(iris_client, WaveformClient)
        assert not isinstance(10, WaveformClient)

    def test_fdsn_issubclass(self):
        assert issubclass(Client, WaveformClient)
        assert not issubclass(str, WaveformClient)

    def test_stream(self):
        st = obspy.read()
        assert isinstance(st, WaveformClient)
        assert issubclass(obspy.Stream, WaveformClient)

    def test_wavebank(self, bingham_dataset):
        wavebank = bingham_dataset.waveform_client
        assert isinstance(wavebank, WaveformClient)
        assert issubclass(WaveBank, WaveformClient)


class TestStationClient:
    def test_fdsn_isinstance(self, iris_client):
        """ ensure the client is an instance of EventClient """
        assert isinstance(iris_client, StationClient)
        assert not isinstance(10, StationClient)

    def test_fdsn_issubclass(self):
        assert issubclass(Client, StationClient)
        assert not issubclass(str, StationClient)

    def test_inventory(self):
        inv = obspy.read_inventory()
        assert isinstance(inv, StationClient)
        assert issubclass(obspy.Inventory, StationClient)
