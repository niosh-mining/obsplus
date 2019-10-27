"""
Conftest specifically for waveform tests.
"""
import numpy as np
import obspy
import pytest


# ----------------------------- helper functions


def gen_func(name):
    def somefunc(self, x="bob"):
        setattr(self, name, x)

    """ generate a dummy method that sets the cache """

    return somefunc


def _change_stats(st, stat_attr, new_values):
    """ change the stats attrs in a waveforms """
    for tr, new_val in zip(st, new_values):
        setattr(tr.stats, stat_attr, new_val)


# ---------------------------- fixtures


@pytest.fixture(scope="class")
def basic_stream_with_gap(waveform_cache):
    """ return a waveforms with a 2 second gap in center, return combined
    waveforms with gaps, first chunk, second chunk """
    st = waveform_cache["default"]
    st1 = st.copy()
    st2 = st.copy()
    t1 = st[0].stats.starttime
    t2 = st[0].stats.endtime
    average = obspy.UTCDateTime((t1.timestamp + t2.timestamp) / 2.0)
    a1 = average - 1
    a2 = average + 1
    # split and recombine
    st1.trim(starttime=t1, endtime=a1)
    st2.trim(starttime=a2, endtime=t2)
    out = st1.copy() + st2.copy()
    out.sort()
    assert len(out) == 6
    gaps = out.get_gaps()
    for gap in gaps:
        assert gap[4] < gap[5]
    return out, st1, st2


@pytest.fixture(scope="class")
def disjointed_stream():
    """ return a waveforms that has parts with no overlaps """
    st = obspy.read()
    st[0].stats.starttime += 3600
    return st


@pytest.fixture(scope="class")
def stream_dict(waveform_cache):
    """ return a dictionary of streams """
    out = {}
    st = waveform_cache["coincidence_tutorial"]
    for var in range(5):
        st = st.copy()
        for tr in st:
            # change starttime
            tr.stats.starttime += 3600 * 24 * var
            # add noise
            tr.data = tr.data.astype(np.float64)
            med = np.median(tr.data)
            tr.data += np.random.rand(len(tr.data)) * med
        out["event_" + str(var)] = st
    return out


@pytest.fixture
def fragmented_stream():
    """ create a waveforms that has been fragemented """
    st = obspy.read()
    # make streams with new stations that are disjointed
    st2 = st.copy()
    for tr in st2:
        tr.stats.station = "BOB"
        tr.data = tr.data[0:100]
    st3 = st.copy()
    for tr in st3:
        tr.stats.station = "BOB"
        tr.stats.starttime += 25
        tr.data = tr.data[2500:]
    return st + st2 + st3
