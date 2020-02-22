"""
Testing utilities for ObsPlus.
"""
import os
from collections import Counter
from contextlib import contextmanager
from os.path import join
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import obspy
import pandas as pd

from obsplus.constants import NSLC, utc_able_type
from obsplus.utils.time import make_time_chunks, to_utc


@contextmanager
def instrument_methods(obj):
    """
    Temporarily instrument an object's methods.

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

    def create_stream(
        self,
        starttime: utc_able_type,
        endtime: utc_able_type,
        seed_ids: Optional[List[str]] = None,
        sampling_rate: Optional[Union[float, int]] = None,
    ) -> obspy.Stream:
        """ create a waveforms from random data """
        t1 = to_utc(starttime)
        t2 = to_utc(endtime)
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
            raise ValueError("something went wrong!")  # pragma: no cover

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


def assert_streams_almost_equal(
    st1: obspy.Stream,
    st2: obspy.Stream,
    basic_stats: bool = True,
    atol: float = 1e-05,
    rtol: float = 1e-08,
    equal_nan: bool = True,
    allow_off_by_one: bool = False,
) -> None:
    """
    Assert that two streams are almost equal else raise helpful exceptions.

    Parameters
    ----------
    st1
        The first stream
    st2
        The second stream
    basic_stats
        If True, only compare basic stats of the streams including:
        network, station, location, channel, starttime, endtime
    atol
        The absolute tolerance parameter
    rtol
        The relative tolerance parameter
    equal_nan
        If True evaluate NaNs as equal
    allow_off_by_one
        If True, allow the arrays and time alignments to be off by one sample.

    Notes
    -----
    See numpy.allclose for description of atol and rtol paramter.

    Raises
    ------
    AssertionError if streams are not about equal.
    """
    kwargs = dict(
        basic_stats=basic_stats,
        atol=atol,
        rtol=rtol,
        equal_nan=equal_nan,
        allow_off_by_one=allow_off_by_one,
    )
    _StreamEqualTester(**kwargs)(st1, st2)


class _StreamEqualTester:
    """
    Simple class for testing if streams are (almost) equal.

    This class is not intended to be used directly, instead use
    :func:`obsplus.utils.testing.assert_streams_almost_equal`.
    """

    skeys = list(NSLC) + ["sampling_rate", "starttime", "endtime"]

    def __init__(
        self,
        basic_stats: bool = True,
        atol: float = 1e-05,
        rtol: float = 1e-08,
        equal_nan: bool = True,
        allow_off_by_one: bool = False,
    ):
        self.basic_stats = basic_stats
        self.atol = atol
        self.rtol = rtol
        self.equal_nan = equal_nan
        self.allow_off_by_one = allow_off_by_one

    def _assert_stats_equal(self, tr1, tr2):
        """ Assert that the stats dicts are almost equal. """
        skeys, basic_stats = self.skeys, self.basic_stats
        sta1 = {x: tr1.stats[x] for x in skeys} if basic_stats else tr1.stats
        sta2 = {x: tr2.stats[x] for x in skeys} if basic_stats else tr2.stats
        if not sta1 == sta2:
            stats_equal = False
            # see if the start and end times are within one sample rate
            if self.allow_off_by_one:
                times = ("starttime", "endtime")
                sta1_new = {i: v for i, v in sta1.items() if i not in times}
                sta2_new = {i: v for i, v in sta2.items() if i not in times}
                if sta1["sampling_rate"] == sta2["sampling_rate"]:
                    sr = sta1["sampling_rate"]
                    t1_diff = abs(sta1["starttime"] - sta2["starttime"])
                    t2_diff = abs(sta1["endtime"] - sta2["starttime"])
                    if t1_diff < sr and t2_diff < sr and sta1_new == sta2_new:
                        stats_equal = True
            if not stats_equal:
                msg = f"Stats are not the same for the traces: \n{sta1}\n{sta2}"
                assert 0, msg

    def _assert_arrays_almost_equal(self, tr1, tr2):
        """ Assert that the data arrays of the traces are almost equal. """
        ars = sorted([tr1.data, tr2.data], key=lambda x: len(x))
        len1, len2 = len(ars[0]), len(ars[1])
        len_diff = len2 - len1
        kwargs = dict(atol=self.atol, rtol=self.rtol, equal_nan=self.equal_nan)
        # check for off by one error
        if len_diff != 0:  # if they aren't equal in len
            close = False
        else:
            close = np.allclose(ars[0], ars[1], **kwargs)
        if not close and self.allow_off_by_one:
            # If the arrays are within 2 samples of each other in length
            if abs(len(ars[0]) - len(ars[1])) <= 2:
                sub_ar = ars[0][1:-1]
                # slide the smaller array over the larger, return true if found
                for i in range(len(ars[1]) - len(ars[0]) + 3):
                    if np.allclose(sub_ar, ars[1][i : i + len(sub_ar)], **kwargs):
                        close = True
                        break
        assert close, "Data of traces are not nearly equal"

    def __call__(self, st1, st2):
        """
        Assert that two streams are almost equal else raise AssertionError.

        Parameters
        ----------
        st1
            The first stream
        st2
            The second stream
        """
        st1.sort(), st2.sort()
        if len(st1) != len(st2):
            assert 0, "streams do not have the same number of traces"
        # iterate each trace and raise if stats and arrays are not almost equal.
        for tr1, tr2 in zip(st1, st2):
            self._assert_stats_equal(tr1, tr2)
            self._assert_arrays_almost_equal(tr1, tr2)
