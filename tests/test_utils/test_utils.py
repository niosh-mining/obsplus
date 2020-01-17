""" tests for various utility functions """
import itertools
import os
import time
from pathlib import Path

import numpy as np
import obspy
import obspy.core.event as ev
import pandas as pd
import pytest

import obsplus
import obsplus.utils.misc
from obsplus.constants import NSLC, NULL_SEED_CODES
from obsplus.utils.misc import (
    yield_obj_parent_attr,
    iter_files,
    getattrs,
    read_file,
    deprecated_callable,
)
from obsplus.utils.pd import filter_index, filter_df
from obsplus.utils.time import to_timestamp


# ------------------------- module level fixtures


class TestIterate:
    def test_none(self):
        """ None should return an empty tuple """
        assert obsplus.utils.misc.iterate(None) == tuple()

    def test_object(self):
        """ A single object should be returned in a tuple """
        assert obsplus.utils.misc.iterate(1) == (1,)

    def test_str(self):
        """ A single string object should be returned as a tuple """
        assert obsplus.utils.misc.iterate("hey") == ("hey",)


class TestReplaceNullSeedCodes:
    """ tests for replacing nulish NSLC codes for various objects. """

    @pytest.fixture
    def null_stream(self,):
        """ return a stream with various nullish nslc codes. """
        st = obspy.read()
        st[0].stats.location = ""
        st[1].stats.channel = "None"
        st[2].stats.network = "null"
        st[0].stats.station = "--"
        return st

    @pytest.fixture
    def null_catalog(self):
        """ create a catalog object, hide some nullish station codes in
        picks and such """

        def make_wid(net="UU", sta="TMU", loc="01", chan="HHZ"):
            kwargs = dict(
                network_code=net, station_code=sta, location_code=loc, channel_code=chan
            )
            wid = ev.WaveformStreamID(**kwargs)
            return wid

        cat = obspy.read_events()
        ev1 = cat[0]
        # make a pick
        picks = []
        for val in NULL_SEED_CODES:
            wid = make_wid(loc=val)
            picks.append(ev.Pick(waveform_id=wid, time=obspy.UTCDateTime()))
        ev1.picks.extend(picks)
        return cat

    @pytest.fixture
    def null_inventory(self):
        """ Create an inventory with various levels of nullish chars. """
        inv = obspy.read_inventory()
        # change the location codes, all other codes are required
        inv[0][0][1].location_code = "--"
        inv[0][0][2].location_code = "None"
        inv[0][1][1].location_code = "nan"
        return inv

    def test_stream(self, null_stream):
        """ ensure all the nullish chars are replaced """
        st = obsplus.utils.misc.replace_null_nlsc_codes(null_stream.copy())
        for tr1, tr2 in zip(null_stream, st):
            for code in NSLC:
                code1 = getattr(tr1.stats, code)
                code2 = getattr(tr2.stats, code)
                if code1 in NULL_SEED_CODES:
                    assert code2 == ""
                else:
                    assert code1 == code2

    def test_catalog(self, null_catalog):
        """ ensure all nullish catalog chars are replaced """
        cat = obsplus.utils.misc.replace_null_nlsc_codes(null_catalog.copy())
        for pick, _, _ in yield_obj_parent_attr(cat, cls=ev.Pick):
            wid = pick.waveform_id
            assert wid.location_code == ""

    def test_inventory(self, null_inventory):
        def _valid_code(code):
            """ return True if the code is valid. """
            return code not in NULL_SEED_CODES

        inv = obsplus.utils.misc.replace_null_nlsc_codes(null_inventory)

        for net in inv:
            assert _valid_code(net.code)
            for sta in net:
                assert _valid_code(sta.code)
                for chan in sta:
                    assert _valid_code(chan.code)
                    assert _valid_code(chan.location_code)


class TestFilterDf:
    @pytest.fixture
    def example_df(self):
        """ create a simple df for testing. Example from Chris Albon. """
        raw_data = {
            "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
            "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
            "age": [42, 52, 36, 24, 73],
            "preTestScore": [4, 24, 31, 2, 3],
            "postTestScore": [25, 94, 57, 62, 70],
        }
        return pd.DataFrame(raw_data, columns=list(raw_data))

    def test_filter_index(self, crandall_dataset):
        """ Tests for filtering index with filter index function. """
        # this is mainly here to test the time filtering, because the bank
        # operations pass this off to the HDF5 kernel.
        index = crandall_dataset.waveform_client.read_index(network="UU")
        t1_ns = int(index["starttime"].astype(int).mean())
        t1 = np.datetime64(t1_ns, "ns")
        t2 = index["endtime"].max()
        kwargs = dict(network="UU", station="*", location="*", channel="*")
        bool_ind = filter_index(index, starttime=t1, endtime=t2, **kwargs)
        assert (~np.logical_not(bool_ind)).any()

    def test_string_basic(self, example_df):
        """ test that specifying a string with no matching works. """
        out = filter_df(example_df, first_name="Jason")
        assert out[0]
        assert not out[1:].any()

    def test_string_matching(self, example_df):
        """ unix style matching should also work. """
        # test *
        out = filter_df(example_df, first_name="J*")
        assert {"Jason", "Jake"} == set(example_df[out].first_name)
        # test ???
        out = filter_df(example_df, first_name="J???")
        assert {"Jake"} == set(example_df[out].first_name)

    def test_str_sequence(self, example_df):
        """ Test str sequences find values in sequence. """
        out = filter_df(example_df, last_name={"Miller", "Jacobson"})
        assert out[:2].all()
        assert not out[2:].any()

    def test_non_str_single_arg(self, example_df):
        """ test that filter index can be used on Non-nslc columns. """
        # test non strings
        out = filter_df(example_df, age=42)
        assert out[0]
        assert not out[1:].any()

    def test_non_str_sequence(self, example_df):
        """ ensure sequences still work for isin style comparisons. """
        out = filter_df(example_df, age={42, 52})
        assert out[:2].all()
        assert not out[2:].any()

    def test_bad_parameter_raises(self, example_df):
        """ ensure passing a parameter that doesn't have a column raises. """
        with pytest.raises(ValueError):
            filter_df(example_df, bad_column=2)


class TestMisc:
    """ misc tests for small utilities """

    @pytest.fixture
    def apply_test_dir(self, tmpdir):
        """ create a test directory for applying functions to files. """
        path = Path(tmpdir)
        with (path / "first_file.txt").open("w") as fi:
            fi.write("hey")
        with (path / "second_file.txt").open("w") as fi:
            fi.write("ho")
        return path

    def test_no_std_out(self, capsys):
        """ ensure print doesn't propagate to std out when suppressed. """
        with obsplus.utils.misc.no_std_out():
            print("whisper")
        # nothing should have made it to stdout to get captured. The
        # output shape seems to be dependent on pytest version (or something).
        capout = capsys.readouterr()
        if isinstance(capout, tuple):
            assert [not x for x in capout]
        else:
            assert not capout.out

    def test_to_timestamp(self):
        """ ensure things are properly converted to timestamps. """
        ts1 = to_timestamp(10, None)
        ts2 = obspy.UTCDateTime(10).timestamp
        assert ts1 == ts2
        on_none = to_timestamp(None, 10)
        assert on_none == ts1 == ts2

    def test_apply_or_skip(self, apply_test_dir):
        """ test applying a function to all files or skipping """
        processed_files = []

        def func(path):
            processed_files.append(path)
            if "second" in path.name:
                raise Exception("I dont want seconds")
            return path

        out = list(obsplus.utils.misc.apply_to_files_or_skip(func, apply_test_dir))
        assert len(processed_files) == 2
        assert len(out) == 1

    def test_getattrs_unused_attr(self):
        """ simple tests for getattrs """
        instance = "instance"
        assert getattrs(instance, ("bob",)) == {"bob": np.NaN}

    def test_read_file_fails(self):
        """ ensure read_file can raise IOError. """

        def raise_value_error(arg):
            raise ValueError("ouch")

        path = "something made up"
        with pytest.raises(IOError):
            read_file(path, (raise_value_error,))

    def test_deprecate_function(self):
        """ Test deprecating function. """

        @deprecated_callable(replacement_str="another_func")
        def func():
            return None

        with pytest.warns(UserWarning) as w:
            func()

        assert len(w.list) == 1
        assert "is deprecated" in str(w.list[0])

    def test_getattrs_none_returns_empty(self):
        """ make sure None returns empty dict"""
        out = getattrs(None, ["bob"])
        assert isinstance(out, dict)
        assert not out


class TestProgressBar:
    """ Tests for progress bar functionality. """

    def test_graceful_progress_fail(self, monkeypatch):
        """ Ensure a progress bar that cant update returns None """
        from progressbar import ProgressBar

        def raise_exception():
            raise Exception

        monkeypatch.setattr(ProgressBar, "start", raise_exception)
        assert obsplus.utils.misc.get_progressbar(100) is None

    def test_simple_progress_bar(self,):
        """ Ensure a simple progress bar can be used. """
        from progressbar import ProgressBar

        bar = obsplus.utils.misc.get_progressbar(max_value=100, min_value=1)
        assert isinstance(bar, ProgressBar)
        bar.update(1)  # if this doesn't raise the test passes

    def test_none_if_min_value_not_met(self):
        """ Bar should return None if the min value isn't met. """
        bar = obsplus.utils.misc.get_progressbar(max_value=1, min_value=100)
        assert bar is None


class TestMD5:
    """ Tests for getting md5 hashes from files. """

    @pytest.fixture(scope="class")
    def directory_md5(self, tmpdir_factory):
        """ Create an MD5 directory for testing. """
        td = Path(tmpdir_factory.mktemp("md5test"))
        with (td / "file1.txt").open("w") as fi:
            fi.write("test1")
        subdir = td / "subdir"
        subdir.mkdir(exist_ok=True, parents=True)
        with (subdir / "file2.txt").open("w") as fi:
            fi.write("test2")
        return td

    @pytest.fixture(scope="class")
    def md5_out(self, directory_md5):
        """ return the md5 of the directory. """
        return obsplus.utils.misc.md5_directory(directory_md5, exclude="*1.txt")

    def test_files_exist(self, md5_out):
        """ make sure the hashes exist for the files and such """
        # the file1.txt should not have been included
        assert len(md5_out) == 1
        assert "file1.txt" not in md5_out


class TestYieldObjectParentAttr:
    """ tests for yielding objects, parents, and attributes. """

    def test_get_origins(self):
        """ A simple test to get origins from the default catalog. """
        cat = obspy.read_events()
        origins1 = [x[0] for x in yield_obj_parent_attr(cat, cls=ev.Origin)]
        origins2 = list(itertools.chain.from_iterable([x.origins for x in cat]))
        assert len(origins1) == len(origins2)

    def test_object_with_slots(self):
        """ Ensure it still works with slots objects. """

        class Slot:
            __slots__ = ("hey", "bob")

            def __init__(self, hey, bob):
                self.hey = hey
                self.bob = bob

        slot = Slot(hey=ev.ResourceIdentifier("bob"), bob="ugh")

        rids = [x[0] for x in yield_obj_parent_attr(slot, ev.ResourceIdentifier)]
        assert len(rids) == 1
        assert str(rids[0]) == "bob"


class TestIterFiles:
    """" Tests for iterating directories of files. """

    sub = {"D": {"C": ".mseed"}, "F": ".json", "G": {"H": ".txt"}}
    file_paths = {"A": ".txt", "B": sub}

    # --- helper functions
    def setup_test_directory(self, some_dict: dict, path: Path):
        for path in self.get_file_paths(some_dict, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as fi:
                fi.write("useful text")

    def get_file_paths(self, some_dict, path):
        """ return expected paths to files. """
        for i, v in some_dict.items():
            if isinstance(v, dict):
                yield from self.get_file_paths(v, path / i)
            else:
                yield path / (i + v)

    # --- fixtures
    @pytest.fixture(scope="class")
    def simple_dir(self, tmp_path_factory):
        path = Path(tmp_path_factory.mktemp("iterfiles"))
        self.setup_test_directory(self.file_paths, path)
        return path

    def test_basic(self, simple_dir):
        """ test basic usage of iterfiles. """
        files = set(self.get_file_paths(self.file_paths, simple_dir))
        out = set((Path(x) for x in iter_files(simple_dir)))
        assert files == out

    def test_one_subdir(self, simple_dir):
        subdirs = simple_dir / "B" / "D"
        out = set(iter_files(subdirs))
        assert len(out) == 1

    def test_multiple_subdirs(self, simple_dir):
        path1 = simple_dir / "B" / "D"
        path2 = simple_dir / "B" / "G"
        out = {Path(x) for x in iter_files([path1, path2])}
        files = self.get_file_paths(self.file_paths, simple_dir)
        expected = {
            x
            for x in files
            if str(x).startswith(str(path1)) or str(x).startswith(str(path2))
        }
        assert out == expected

    def test_extention(self, simple_dir):
        out = set(iter_files(simple_dir, ext=".txt"))
        for val in out:
            assert val.endswith(".txt")

    def test_mtime(self, simple_dir):
        files = list(self.get_file_paths(self.file_paths, simple_dir))
        # set the first file mtime in future
        now = time.time()
        first_file = files[0]
        os.utime(first_file, (now + 10, now + 10))
        # get output make sure it only returned first file
        out = list(iter_files(simple_dir, mtime=now))
        assert len(out) == 1
        assert Path(out[0]) == first_file
