"""
Infrastructure for caching waveforms using an obsplus bank
"""
from collections import defaultdict
from functools import reduce
from operator import add
from typing import Callable

import obspy
import pandas as pd

import obsplus
from obsplus.utils.pd import filter_index

processor_type = Callable[[obspy.Stream], obspy.Stream]

import obspy.core.event.event


class WaveCache:
    """
    Wrapper class for caching streams from a client.

    Current only supports :class:`~obsplus.WaveBank`.

    Parameters
    ----------
    waveform_client
        A client that will return waveforms.
    """

    def __init__(self, waveform_client: obsplus.WaveBank):
        assert hasattr(waveform_client, "read_index")
        # dicts that will be used as caches for processed/unprocessed streams
        self._stream_cache = {}
        self._processed_cache = {}
        # list of callables that should take a single waveforms and return
        self._stream_processors = defaultdict(list)
        # load all of the bank index into memory
        self.bank = waveform_client
        self.index = waveform_client.read_index()
        # make a column of callables for getting waveforms
        self.index["st_call"] = self.index.apply(self._make_st_calls, axis=1)
        # get a column that identifies rows that have the same resource
        self.index.rename(columns={"path": "unique_key"}, inplace=True)

    def _make_st_calls(self, row):
        """make waveforms callables"""

        def _func(key=row.path):
            if key not in self._processed_cache:
                # get the waveforms and process, add to process cache
                if key in self._stream_cache:
                    st = self._stream_cache[key]
                else:
                    path = self.bank.bank_path + key
                    st = obspy.read(path, format=self.bank.format)
                    self._stream_cache[key] = st
                self._processed_cache[key] = self._apply_processors(st)
            return self._processed_cache[key]

        return _func

    # --- get waveforms

    def get_waveforms(
        self,
        network=None,
        station=None,
        location=None,
        channel=None,
        starttime=None,
        endtime=None,
    ) -> obspy.Stream:
        """
        Get waveforms from the cache, read from disk and cache if needed.

        See obplus.WaveBank.get_waveforms for param descriptions.
        """
        filt = filter_index(
            self.index, network, station, location, channel, starttime, endtime
        )
        ser = self.index[filt].set_index("unique_key")["st_call"]
        # drop duplicates
        ser = ser[~ser.index.duplicated()]
        # no waveforms found, return empty waveforms
        if not len(ser):
            return obspy.Stream()

        st = reduce(add, (x() for x in ser))
        if starttime is not None or endtime is not None:
            # use start/endtime or set far out constants
            starttime = starttime or 0
            endtime = endtime or 32503680000
            return st.trim(starttime=starttime, endtime=endtime)
        else:
            return st

    def add_waveforms(self, st):
        """add waveforms to cache"""
        # get unique id and add to cache
        uid = id(st)
        self._stream_cache[uid] = st
        new_df = pd.DataFrame([self._get_trace_info(tr, uid) for tr in st])
        self.index = pd.concat([self.index, new_df], axis=0, sort=True)

    def _get_trace_info(self, tr, uid):
        """return a dict of info about particular trace"""
        net, sta, loc, chan = tr.id.split(".")
        t1, t2 = tr.stats.starttime.timestamp, tr.stats.endtime.timestamp

        def _func(uid=uid):
            return self._stream_cache.get(uid, obspy.Stream())

        return dict(
            network=net,
            station=sta,
            location=loc,
            channel=chan,
            starttime=t1,
            endtime=t2,
            unique_key=uid,
            st_call=_func,
        )

    def _apply_processors(self, st):
        """apply the current processors to the waveforms"""
        st = st.copy().merge(1)  # do not modify raw data
        for priority in sorted(self._stream_processors.keys()):
            for func in self._stream_processors[priority]:
                st = func(st) or st  # handles in-place operators as well
        return st

    def add_processor(self, stream_processor: processor_type, priority=1):
        """
        Add a processor to the waveforms cache.

        A Processor should be a callable that takes as single waveforms as the
        only argument and returns a waveforms. It is recommended you do not copy
        the waveforms as it will be copied and merged once before applying
        processors.

        Parameters
        ----------
        stream_processor
            A callable for processing streams.
        priority
            An integer to determine order in which processors are applied.
            Lower numbers are applied first.
        """
        self._processed_cache.clear()  # clear processed cache
        self._stream_processors[priority].append(stream_processor)

    def clear_processors(self):
        """remove all waveforms processors and reset cache"""
        self._processed_cache.clear()
        self._stream_processors.clear()
