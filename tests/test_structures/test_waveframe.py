"""
Tests for the WaveFrame class.
"""
import obspy
import pytest

from obsplus import WaveFrame


@pytest.fixture
def waveframe():
    """ Create a basic WaveFrame from default stream. """
    st = obspy.read()
    return WaveFrame.from_stream(st)


class TestToFromStream:
    """ Tests for converting a stream to a WaveFrame. """

    def test_type(self, waveframe):
        assert isinstance(waveframe, WaveFrame)

    def test_to_stream(self, waveframe):
        st = waveframe.to_stream()
        assert st == obspy.read()
