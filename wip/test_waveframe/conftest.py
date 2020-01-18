"""
Waveframe sp
"""

import obspy
import pytest

from obsplus import WaveFrame


def _make_st_no_response():
    """ Get a copy of the default trace, remove response. """
    st = obspy.read()
    # drop response for easier stats dtypes
    for tr in st:
        tr.stats.pop("response", None)
    return st


@pytest.fixture
def st_no_response():
    """ Get a copy of the default trace, remove response. """
    return _make_st_no_response()


@pytest.fixture
def stream_wf(st_no_response) -> WaveFrame:
    """ Create a basic WaveFrame from default stream. """
    return WaveFrame.load_example_wf()


@pytest.fixture
def waveframe_gap(st_no_response) -> WaveFrame:
    """
    Create a waveframe with a 1 second gap in the middle.
    Also shorten last trace.
    """
    st1, st2 = st_no_response.copy(), st_no_response.copy()
    for tr in st2:
        tr.stats.starttime = st1[0].stats.endtime + 1
    st2[-1].data = st2[-1].data[:-20]
    wf = WaveFrame.from_stream(st1 + st2)
    assert isinstance(wf, WaveFrame)
    assert len(wf) == 6
    return wf
