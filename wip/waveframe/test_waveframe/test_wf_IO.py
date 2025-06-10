import obspy

from obsplus import WaveFrame


class TestToFromStream:
    """Tests for converting a stream to a WaveFrame."""

    def test_type(self, stream_wf):
        assert isinstance(stream_wf, WaveFrame)

    def test_to_stream(self, stream_wf, st_no_response):
        st = stream_wf.to_stream()
        assert isinstance(st, obspy.Stream)
        assert st == st_no_response
