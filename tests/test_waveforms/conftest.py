""""""
import numpy as np
import obspy
import pytest
from obsplus import obspy_to_array


# ----------------------------- helper functions


def gen_func(name):
    def somefunc(self, x="bob"):
        setattr(self, name, x)

    """ generate a dummy method that sets the cache """

    return somefunc


def make_search_window_array(embed_index, template_len, search_window_len):
    """ return an array of random search space with the normalized
    default trace object embbed in it """
    st = obspy.read()
    ei = embed_index
    tl = template_len
    swl = search_window_len
    for num, tr in enumerate(st):
        event_data = (tr.data / np.std(tr.data))[:tl]
        replace_data = np.random.rand(swl)
        replace_data[ei : ei + tl] = event_data
        tr.data = replace_data
        tr.stats.starttime += 5000  # add arbitrary to start time
    # st_dict = {str(st[0].stats.starttime): st}
    return obspy_to_array(st).copy(deep=True)


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
def default_array():
    """ the basic waveforms turned into an array """
    st = obspy.read()
    return obspy_to_array(st)


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


@pytest.fixture(scope="class")
def data_array_from_dict(stream_dict):
    """ feed the waveforms dict into the stream_dict2dataset functions """
    return obspy_to_array(stream_dict)


@pytest.fixture(scope="class")
def bingham_dar(bingham_stream_dict):
    """ build a data array from the bingham waveforms dictionary """
    return obspy_to_array(bingham_stream_dict)


@pytest.fixture(scope="class")
def dar_attached_picks(bingham_dataset, bingham_dar):
    """ return a data array from the waveforms after attaching cat  """
    cat = bingham_dataset.event_client.get_events()
    bingham_dar.ops.attach_events(cat)
    assert not bingham_dar.isnull().any(), "nulls found in data array"
    return bingham_dar


# @pytest.fixture(scope="class", params=list(pytest.waveforms.keys))
# def stream(request):
#     """ each of the streams in test_data/file_paths"""
#     return pytest.waveforms[request.param]


@pytest.fixture(scope="class")
def dar_disjoint_seeds():
    """ make a waveforms dict with various channels and networks, but no common
     ones across trace ids """
    # waveforms 0, default waveforms
    streams = {0: obspy.read()}
    # waveforms 1, 3 defaults
    st1 = obspy.read()
    st2 = obspy.read()
    st3 = obspy.read()
    for tr in st2:
        tr.stats.station = "BOB"
    for tr in st3:
        tr.stats.network = "UU"
    streams[1] = st1 + st2 + st3
    return obspy_to_array(streams)


@pytest.fixture(scope="session")
def many_sid_array():
    """ create a data array with many different station ids """
    # first create, and mutate, several waveforms objects
    st_list = [obspy.read() for _ in range(5)]
    # modify stats
    _change_stats(st_list[0], "network", ["UU"] * 3)
    _change_stats(st_list[1], "station", ["RBOB"] * 3)
    _change_stats(st_list[2], "location", ["01"] * 3)
    _change_stats(st_list[3], "channel", ["HN" + x for x in "ENZ"])
    # conglomerate into one waveforms
    st = obspy.Stream()
    for st_ in st_list:
        st += st_
    # make dict and return data array
    st_dict = {key: st.copy() for key in range(10)}
    return obspy_to_array(st_dict)


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
