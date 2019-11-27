import obspy

from obsplus import WaveFrame


class TestToFromStream:
    """ Tests for converting a stream to a WaveFrame. """

    def test_type(self, waveframe_from_stream):
        assert isinstance(waveframe_from_stream, WaveFrame)

    def test_to_stream(self, waveframe_from_stream, st_no_response):
        st = waveframe_from_stream.to_stream()
        assert isinstance(st, obspy.Stream)
        assert st == st_no_response
