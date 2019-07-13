"""
Ensure the interface isinstance and issubclass methods work
"""
import obspy
import pytest
from obspy.clients.fdsn.client import Client, FDSNException
from progressbar import ProgressBar as ProgBar

from obsplus import EventBank, WaveBank
from obsplus.interfaces import EventClient, WaveformClient, StationClient, ProgressBar


# fixtures


@pytest.fixture(scope="session")
def iris_client():
    """ return the IRIS client """
    try:
        return Client()
    except FDSNException:
        pytest.skip("could not connect to webservice")


class TestEventClient:
    def test_fdsn_isinstance(self, iris_client):
        """ ensure the client is an instance of EventClient """
        assert isinstance(iris_client, EventClient)
        breakpoint()
        isinstance(10, EventClient)

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

    def test_cls_with_getattr(self, bingham_dataset):
        """
        A class that returns objects for the appropriate attributes
        with getattr rather than explicit methods should still be a
        subclass/instances.
        """
        wavebank = bingham_dataset.waveform_client

        class MockWaveform:
            def __getattr__(self, item):
                if item == "get_waveforms":
                    return wavebank.get_waveforms
                raise AttributeError(f"No such attribute {item}")

        assert isinstance(MockWaveform(), WaveformClient)


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


class TestBar:
    """ Tests the progressbar interface. """

    def test_progressbar_isinstance(self):
        """ Ensure the ProgressBar2 ProgressBar is an instance. """
        assert issubclass(ProgBar, ProgressBar)

    def test_custom_progress_bar(self):
        """ Ensure custom progress bar works as well. """

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
