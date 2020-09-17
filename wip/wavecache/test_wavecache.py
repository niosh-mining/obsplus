"""
Test the waveform caching
"""

import obspy
import pytest
from obsplus import WaveBank
from obsplus.structures.wavecache import WaveCache


# --- fixtures


@pytest.fixture
def wavecache(bingham_dataset):
    """ return a wavecache of bingham_test archive """
    bank = WaveBank(bingham_dataset.waveform_client)
    return WaveCache(bank)


class TestBasics:
    def test_load_waveforms(self, wavecache):
        """ ensure cache loads waveforms """
        t1 = obspy.UTCDateTime(wavecache.index.starttime.iloc[0])
        st = wavecache.get_waveforms(starttime=t1, endtime=t1 + 12)
        assert isinstance(st, obspy.Stream)

    def test_chached_waveforms(self, wavecache):
        """ ensure the same waveforms is returned on multiple calls """
        t1 = obspy.UTCDateTime(wavecache.index.starttime.iloc[0])
        st1 = wavecache.get_waveforms(starttime=t1, endtime=t1 + 12)
        st2 = wavecache.get_waveforms(starttime=t1, endtime=t1 + 12)
        assert st1 == st2
        # ensure the traces were not reloaded
        assert {id(tr) for tr in st1} == {id(tr) for tr in st2}

    def test_add_waveforms(self, wavecache):
        """ ensure waveforms can be added and returned """
        st = obspy.read()
        wavecache.add_waveforms(st)
        st2 = wavecache.get_waveforms(station="RJOB")
        assert st2 is st

    def test_add_processor(self, wavecache):
        """ensure when a processor is added it runs on the streams, and all
        processors can be cleared."""
        inds = [{}, {}]
        index = 0

        # define waveforms processor and add to cache
        def _func(st):
            for tr in st:
                inds[index][(id(tr))] = tr
                tr.new_attr = "its here"
            return st

        wavecache.add_processor(_func)

        # get waveforms and assert new attr is found
        for tr in wavecache.get_waveforms():
            assert hasattr(tr, "new_attr")
            assert tr.new_attr == "its here"
            assert id(tr) in inds[0]

        # ensure adding another processor clears out processed cache
        index = 1

        def _another_func(st):
            return st

        wavecache.add_processor(_another_func)

        for tr in wavecache.get_waveforms():
            assert hasattr(tr, "new_attr")
            assert tr.new_attr == "its here"
            assert id(tr) not in inds[0]
            assert id(tr) in inds[1]

        # clear cache and ensure new attr is no longer there
        wavecache.clear_processors()
        for tr in wavecache.get_waveforms():
            assert not hasattr(tr, "new_attr")

    def test_waveforms_are_sliced(self, wavecache):
        """ensure if the starttime/endtime are specified the waveforms are
        trimmed to desired time."""
        # get time of first trace, set trim times to one second in
        st = wavecache.get_waveforms()
        stats = st[0].stats
        t1, t2 = stats.starttime + 1, stats.endtime - 1
        sr = stats.sampling_rate
        for tr in wavecache.get_waveforms(starttime=t1, endtime=t2):
            assert abs(tr.stats.starttime - t1) < (2.0 / sr)
