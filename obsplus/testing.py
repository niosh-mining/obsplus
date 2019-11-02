"""
Simple utilities used for testing.
"""
import os
from collections import Counter
from contextlib import contextmanager
from os.path import join
from pathlib import Path

import numpy as np
import obspy
import pandas as pd

from obsplus.utils.time import make_time_chunks


@contextmanager
def instrument_methods(obj):
    """
    Temporarily instrument an objects methods.

    This allows the calls to each of the objects methods to be counted.
    """
    old_methods = {}
    counter = Counter()
    setattr(obj, "_counter", counter)

    for attr in dir(obj):
        # skip dunders
        if attr.startswith("__"):
            continue
        method = getattr(obj, attr, None)
        # skip anything that isnt callable
        if not callable(method):
            continue
        # append old method to old_methods and create new method
        old_methods[attr] = method

        def func(*args, __method_name=attr, **kwargs):
            counter.update({__method_name: 1})
            return old_methods[__method_name](*args, **kwargs)

        setattr(obj, attr, func)

    # yield monkey patched object
    yield obj
    # reset methods
    for attr, method in old_methods.items():
        setattr(obj, attr, method)
    # delete counter
    delattr(obj, "_counter")


class ArchiveDirectory:
    """ class for creating a simple archive """

    def __init__(
        self,
        path,
        starttime=None,
        endtime=None,
        sampling_rate=1,
        duration=3600,
        overlap=0,
        gaps=None,
        seed_ids=("TA.M17A..VHZ", "TA.BOB..VHZ"),
    ):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        self.starttime = starttime
        self.endtime = endtime
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.overlap = overlap
        self.seed_ids = seed_ids
        self.gaps = gaps

    def create_stream(self, starttime, endtime, seed_ids=None, sampling_rate=None):
        """ create a waveforms from random data """
        t1 = obspy.UTCDateTime(starttime)
        t2 = obspy.UTCDateTime(endtime)
        sr = sampling_rate or self.sampling_rate
        ar_len = int((t2.timestamp - t1.timestamp) * sr)
        st = obspy.Stream()
        for seed in seed_ids or self.seed_ids:
            n, s, l, c = seed.split(".")
            meta = {
                "sampling_rate": sr,
                "starttime": t1,
                "network": n,
                "station": s,
                "location": l,
                "channel": c,
            }
            data = np.random.randn(ar_len)
            tr = obspy.Trace(data=data, header=meta)
            st.append(tr)
        return st

    def get_gap_stream(self, t1, t2, gaps):
        """ return streams with gaps in it """
        assert len(gaps) == 1
        gap = gaps.iloc[0]
        ts1, ts2 = t1.timestamp, t2.timestamp
        # if gap covers time completely
        if gap.start <= ts1 and gap.end >= ts2:
            raise ValueError("gapped out")
        # if gap is contained by time frame
        elif gap.start > ts1 and gap.end < ts2:
            st1 = self.create_stream(ts1, gap.start)
            st2 = self.create_stream(gap.end, ts2)
            return st1 + st2
        # if gap only effects endtime
        elif ts1 < gap.start < ts2 <= gap.end:
            return self.create_stream(ts1, gap.start)
        # if gap only effects starttime
        elif gap.start <= ts1 < gap.end < ts2:
            return self.create_stream(gap.end, ts2)
        else:  # should not reach here
            raise ValueError("something went very wrong!")

    def create_directory(self):
        """ create the directory with gaps in it """
        # get a dataframe of the gaps
        if self.gaps is not None:
            df = pd.DataFrame(self.gaps, columns=["start", "end"])
            df["start"] = df["start"].apply(lambda x: x.timestamp)
            df["end"] = df["end"].apply(lambda x: x.timestamp)
        else:
            df = pd.DataFrame(columns=["start", "end"])

        assert self.starttime and self.endtime, "needs defined times"
        for t1, t2 in make_time_chunks(
            self.starttime, self.endtime, self.duration, self.overlap
        ):
            # figure out of this time lies in a gap
            gap = df[~((df.start >= t2) | (df.end <= t1))]
            if not gap.empty:
                try:
                    st = self.get_gap_stream(t1, t2, gap)
                except ValueError:
                    continue
            else:
                st = self.create_stream(t1, t2)
            finame = str(t1).split(".")[0].replace(":", "-") + ".mseed"
            path = join(self.path, finame)
            st.write(path, "mseed")

    def create_directory_from_bulk_args(self, bulk_args):
        """ Create a directory from bulk waveform arguments """
        # ensure directory exists
        path = Path(self.path)
        path.mkdir(exist_ok=True, parents=True)
        for (net, sta, loc, chan, t1, t2) in bulk_args:
            nslc = ".".join([net, sta, loc, chan])
            st = self.create_stream(t1, t2, (nslc,))
            time_name = str(t1).split(".")[0].replace(":", "-") + ".mseed"
            save_name = path / f"{net}_{sta}_{time_name}"
            st.write(str(save_name), "mseed")
