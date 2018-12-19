"""
Tests for the bank-specific utilities
"""
import os
import tempfile

import obspy
import pytest

from obsplus.bank.utils import _summarize_trace, _summarize_event, _try_read_stream


@pytest.fixture(scope="module")
def text_file():
    """ save some text to file (note a waveforms) return path """
    with tempfile.NamedTemporaryFile("w") as tf:
        tf.write("some text, not a waveforms")
        tf.flush()
        tf.seek(0)
        yield tf.name


class TestStreamPathStructure:
    """ tests for using the PathStructure for Streams """

    structs = [
        "{year}/{month}/{day}",
        "{network}/{station}/{year}/{month}/{day}",
        "{year}/{julday}/{station}",
        "{network}/{station}/{location}/{julday}",
        "{network}/{station}/{location}/{channel}/{year}/{julday}/{hour}/{minute}",
    ]
    stream = obspy.read()

    # fixtures
    @pytest.fixture(scope="class", params=structs)
    def struct_string(self, request):
        """ return the structure strings associated with class"""
        return request.param

    @pytest.fixture(scope="class", params=stream)
    def trace(self, request):
        """ return a trace from the waveforms"""
        return request.param

    @pytest.fixture(scope="class")
    def output(self, struct_string, trace):
        """ init a bank_structure class from the structure strings """
        return _summarize_trace(trace, path_struct=struct_string)

    # general tests

    def test_output(self, output):
        """ test that a dict was returned with required keys when bank
        is called"""
        assert isinstance(output, dict)
        assert isinstance(output["path"], str)

    # specific tests
    def test_trace_path(self):
        """ test the basics of trace conversions """
        struc = "waveforms/{year}/{month}/{day}/{network}/{station}/{channel}"
        tr = obspy.read()[0]
        expected = "waveforms/2009/08/24/BW/RJOB/EHZ/2009-08-24T00-20-03.mseed"
        assert _summarize_trace(tr, path_struct=struc)["path"] == expected


class TestEventPathStructure:
    """ tests for event structures """

    def test_basic(self):
        ev = obspy.read_events()[0]
        expected = "2012/04/04/2012-04-04T14-21-42_00041.xml"
        assert _summarize_event(ev)["path"] == expected


#
# class TestInventoryStructure:
#     """ test for stations path """
#
#     def test_basic(self):
#         path = PathStructure()
#         # get an stations with 1 channel
#         inv = obspy.read_inventory()
#         chan = inv.get_contents()['channel'][0]
#         n, s, l, c = chan.split('.')
#         inv2 = inv.select(network=n, station=s, location=l, channel=c)
#         #
#         out_path = path(inv2)
#
class TestReadStream:
    """ test the read waveforms function """

    # fixtures
    @pytest.fixture()
    def stream_file(self):
        """ save the default waveforms to disk return path """
        name = "deleteme.mseed"
        st = obspy.read()
        st.write(name, "mseed")
        yield name
        os.remove(name)

    def test_bad_returns_none(self, text_file):
        """ make sure bad file returns None """
        with pytest.warns(UserWarning) as warn:
            out = _try_read_stream(text_file)
        assert len(warn)
        expected_str = "obspy failed to read"
        assert any([expected_str in str(w.message) for w in warn])
        assert out is None

    def test_try_read_stream(self, stream_file):
        """ make sure the waveforms file can be read in """
        st = _try_read_stream(stream_file)
        assert isinstance(st, obspy.Stream)
