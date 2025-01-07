"""
Tests for the bank-specific utilities
"""

import os
import tempfile
from typing import ClassVar

import numpy as np
import obspy
import pandas as pd
import pytest
from obsplus.constants import NSLC
from obsplus.utils.bank import (
    _summarize_trace,
    _try_read_stream,
    summarize_generic_stream,
)
from obsplus.utils.events import _summarize_event
from obsplus.utils.mseed import summarize_mseed
from obspy import UTCDateTime


@pytest.fixture(scope="module")
def text_file():
    """Save some text to file (note a waveforms) return path"""
    with tempfile.NamedTemporaryFile("w") as tf:
        tf.write("some text, not a waveforms")
        tf.flush()
        tf.seek(0)
        yield tf.name


class TestStreamPathStructure:
    """tests for using the PathStructure for Streams"""

    structs: ClassVar = [
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
        """Return the structure strings associated with class"""
        return request.param

    @pytest.fixture(scope="class")
    def output(self, struct_string, waveform_cache_trace):
        """Init a bank_structure class from the structure strings"""
        return _summarize_trace(waveform_cache_trace, path_struct=struct_string)

    # general tests
    def test_output(self, output):
        """Test that a dict was returned with required keys."""
        assert isinstance(output, dict)
        assert isinstance(output["path"], str)

    # specific tests
    def test_trace_path(self):
        """Test the basics of trace conversions"""
        struc = "waveforms/{year}/{month}/{day}/{network}/{station}/{channel}"
        tr = obspy.read()[0]
        expected = "waveforms/2009/08/24/BW/RJOB/EHZ/2009-08-24T00-20-03.mseed"
        # expected_str = _expected.replace('/', os.sep)
        out = _summarize_trace(tr, path_struct=struc)["path"]
        assert out.replace(os.sep, "/") == expected


class TestEventPathStructure:
    """tests for event structures"""

    def test_basic(self):
        """Simple sanity check."""
        ev = obspy.read_events()[0]
        expected = "2012/04/04/2012-04-04T14-21-42_00041.xml"
        out = _summarize_event(ev)["path"]
        assert out.replace(os.sep, "/") == expected


class TestReadStream:
    """Test the read waveforms function."""

    # fixtures
    @pytest.fixture()
    def stream_file(self):
        """Save the default waveforms to disk return path"""
        name = "deleteme.mseed"
        st = obspy.read()
        st.write(name, "mseed")
        yield name
        os.remove(name)

    def test_bad_returns_none(self, text_file):
        """Make sure bad file returns None"""
        with pytest.warns(UserWarning) as warn:
            out = _try_read_stream(text_file)
        assert len(warn)
        expected_str = "obspy failed to read"
        assert any([expected_str in str(w.message) for w in warn])
        assert out is None

    def test_try_read_stream(self, stream_file):
        """Make sure the waveforms file can be read in"""
        st = _try_read_stream(stream_file)
        assert isinstance(st, obspy.Stream)


class TestSummarizeStreams:
    """tests for summarizing streams."""

    start = UTCDateTime("2017-09-20T01-00-00")
    end = UTCDateTime("2017-09-20T02-00-00")
    gap_start = UTCDateTime("2017-09-20T01-25-35")
    gap_end = UTCDateTime("2017-09-20T01-25-40")

    def clean_dataframe(self, df):
        """Function to fix some common issues with the dataframe."""
        for id_code in NSLC:
            df[id_code] = (
                df[id_code].astype(str).str.replace("b'", "").str.replace("'", "")
            )
        for time_col in ["starttime", "endtime"]:
            df[time_col] = df[time_col].astype("datetime64[ns]")
        return df[sorted(df.columns)]

    @pytest.fixture
    def gappy_stream(self):
        """Create a very simple mseed with one gap, return it."""
        stats = dict(
            network="UU",
            station="ELU",
            location="01",
            channel="ELZ",
            sampling_rate=1,
            starttime=self.start,
        )
        rng = np.random.default_rng(30)
        len1 = int(self.gap_start - self.start)
        # create first trace
        ar1 = rng.random(len1)
        tr1 = obspy.Trace(data=ar1, header=stats)
        assert tr1.stats.endtime <= self.gap_start
        # create second trace
        len2 = int(self.end - self.gap_end)
        ar2 = rng.random(len2)
        stats2 = dict(stats)
        stats2.update({"starttime": self.gap_end})
        tr2 = obspy.Trace(data=ar2, header=stats2)
        # assemble traces make sure gap is there
        assert tr2.stats.starttime >= self.gap_end
        st = obspy.Stream(traces=[tr1, tr2])
        gaps = st.get_gaps()
        assert len(gaps) == 1
        return st

    @pytest.fixture
    def gappy_mseed_path(self, gappy_stream, tmp_path):
        """Return a path to the saved mseed file with gaps."""
        out_path = tmp_path / "out.mseed"
        gappy_stream.write(str(out_path), format="mseed")
        return out_path

    def test_summarize_mseed(self, gappy_stream, gappy_mseed_path):
        """
        Summarize mseed should return the same answer as the generic
        summary function.
        """
        summary_1 = summarize_mseed(str(gappy_mseed_path))
        df1 = self.clean_dataframe(pd.DataFrame(summary_1))
        summary_2 = summarize_generic_stream(str(gappy_mseed_path))
        df2 = self.clean_dataframe(pd.DataFrame(summary_2))
        assert len(df1) == len(df2)
        assert (df1 == df2).all().all()
