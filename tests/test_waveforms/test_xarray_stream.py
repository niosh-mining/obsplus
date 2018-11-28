"""
Tests for converting streams to xarray objects.
"""

import copy
import os
import tempfile

import numpy as np
import obspy
import pandas as pd
import pytest
import scipy
import xarray as xr

import obsplus
from obsplus import obspy_to_array_dict, obspy_to_array
from obsplus.waveforms.xarray import netcdf2array
from obsplus.waveforms.xarray.aggregate import aggregate, bin_array
from obsplus.waveforms.xarray.io import read_pickle
from obsplus.waveforms.xarray.signal import array_irfft, array_rfft
from obsplus.waveforms.xarray.utils import get_nslc_df, sel_sid, pad_time

rand = np.random.RandomState(13)


# ------------------------- test utilities


def starttimes_consistent(dar, tolerance=0.0001):
    """ Return True if the starttimes in the data array are consistent with
    those in the attrs """

    for eve_id, stat_dict in dar.attrs["stats"].items():
        for chan_id, stats in stat_dict.items():
            dd = dar.sel(seed_id=chan_id, stream_id=eve_id)
            starttime1 = float(dd["starttime"])
            starttime2 = stats.starttime.timestamp
            if abs(starttime1 - starttime2) > tolerance:
                return False
    return True


# ------------------------ tests


class TestWaveform2ArrayDict:
    """ tests for the waveforms to array dict functions """

    # Fixtures that return array dicts with only 1 sampling rate
    homogeneous_fixtures = [
        "array_dict_from_stream_dict",
        "default_array_dict_from_stream",
        "array_dict_from_trace_dict",
        "array_dict_from_trace",
    ]

    # fixtures
    @pytest.fixture(scope="class")
    def default_array_dict_from_stream(self):
        """ return the default waveforms converted to an array dict """
        return obspy_to_array_dict(obspy.read())

    @pytest.fixture(scope="class")
    def array_dict_from_stream_dict(self, stream_dict):
        out = dict(stream_dict)
        out["new"] = obspy.read()[0]  # add a trace to waveforms dict
        return obspy_to_array_dict(stream_dict)

    @pytest.fixture(scope="class")
    def array_dict_from_trace_dict(self):
        """ create an array dict from a trace dict """
        st = obspy.read()
        return obspy_to_array_dict({tr.id: tr for tr in st})

    @pytest.fixture(scope="class")
    def array_dict_from_trace(self):
        """ create an array dict from a trace """
        return obspy_to_array_dict({1: obspy.read()[0]})

    @pytest.fixture(scope="class")
    def array_dict_different_sr(self):
        """ return a data_array dict made from a waveforms with different sample
        rates"""
        st = obspy.read()
        out = {}
        for num, tr in enumerate(st):
            new_sr = num * tr.stats.sampling_rate + 100
            tr.stats["sampling_rate"] = new_sr
            out[num] = obspy.Stream(traces=[tr])
        return obspy_to_array_dict(out)

    # meta fixtures
    @pytest.fixture(scope="class", params=homogeneous_fixtures)
    def homogeneous_array_dict(self, request):
        """ collect all homogeneous fixtures """
        return request.getfixturevalue(request.param)

    # specific tests
    def test_different_sample_rates(self, array_dict_different_sr):
        """ ensure all the sample rates is represented """
        assert len(array_dict_different_sr) == 3
        for item, val in array_dict_different_sr.items():
            assert isinstance(item, int)
            assert isinstance(val, xr.DataArray)

    def test_one_trace(self):
        """ ensure a single trace is valid input """
        tr = obspy.read()[0]
        out = obspy_to_array_dict(tr)
        assert isinstance(out, dict)
        assert len(out) and 100 in out
        dar = out[100]
        assert max(dar.values.shape) == len(tr.data)

    # homogeneous tests
    def test_default_array_is_len_1(self, homogeneous_array_dict):
        """ since the default array only has 1 sampling rate it should produce
        a dict with only one key """
        assert isinstance(homogeneous_array_dict, dict)
        assert len(homogeneous_array_dict) == 1
        # ensure each element is a data array
        for item, value in homogeneous_array_dict.items():
            assert isinstance(value, xr.DataArray)


class TestStreamDict2DataArray:

    # helper functions
    def trim_end(self, st, samps):
        """ trim samps from end of each trace in st """
        st = st.copy()
        for tr in st:
            tr.data = tr.data[:-samps]
        return st

    # fixture
    @pytest.fixture
    def stream_dict_short_len(self):
        """ create a waveforms dict with different lengths """
        st = obspy.read()
        st_bad = st.copy()
        t2 = st_bad[0].stats.endtime
        st_bad = st_bad.slice(endtime=t2 - 10)
        return {1: st.copy(), 2: st_bad, 3: st.copy()}

    @pytest.fixture
    def st_dict_close_lens(self):
        """ create a waveforms dict with streams that are almost the same len"""
        st = obspy.read()
        return {0: st, 1: self.trim_end(st, 10), 2: self.trim_end(st, 15)}

    # tests
    def test_type(self, data_array_from_dict):
        """ ensure the type is correct """
        assert isinstance(data_array_from_dict, xr.DataArray)

    def test_dimensions(self, data_array_from_dict):
        """ ensure there are 3 dimensions """
        assert len(data_array_from_dict.dims) == 3
        assert ("stream_id", "seed_id", "time") == data_array_from_dict.dims

    def test_data_survived(self, data_array_from_dict, stream_dict):
        """ ensure the data did not get changed from conversion """
        for key, st in stream_dict.items():
            for tr in st:
                seed_id = tr.id
                ar = data_array_from_dict.loc[key, seed_id, :]
                assert np.allclose(ar, tr.data)

    def test_attrs_survived(self, data_array_from_dict):
        """ ensure attrs came out """
        assert data_array_from_dict.attrs

    def test_starttime_coord(self, data_array_from_dict):
        """ ensure the starttime is found in the DataArray """
        dar = data_array_from_dict
        assert "starttime" in dar.coords

    def test_bingham_data_array(self, bingham_dar, bingham_stream_dict):
        """ ensure the bingham data array does not contain NaNs and has
        correct starttimes """
        assert not bingham_dar.isnull().any()
        assert set(bingham_dar.stream_id.values) == set(bingham_stream_dict)
        assert starttimes_consistent(bingham_dar, tolerance=0.02)

        for ev_id, st in bingham_stream_dict.items():
            stats_dict = bingham_dar.attrs["stats"][ev_id]
            for seed_id, stats in stats_dict.items():
                time = stats_dict[seed_id].starttime.timestamp
                dd = bingham_dar.sel(seed_id=seed_id, stream_id=ev_id)
                ar_time = float(dd.starttime.values)
                assert abs(time - ar_time) < 1

    def test_disjoint_seed_ids(self, dar_disjoint_seeds):
        """ test that data arrays with disjoint seed ids still return some
        values """
        dar = dar_disjoint_seeds
        assert len(dar.time.values), " empty time dimension "

    def test_stream_processor(self):
        """ ensure a waveforms processor gets called """
        state = []

        def stream_proc(stream):
            state.append(stream)
            return stream

        obspy_to_array(obspy.read(), stream_processor=stream_proc)
        assert len(state)

    def test_empty_stream(self):
        """ test some empty streams as inputs """
        stream_dict = {0: obspy.read(), 1: obspy.read(), 2: obspy.Stream()}
        dar = obspy_to_array(stream_dict)
        assert isinstance(dar, xr.DataArray)
        # empty waveforms should not have been added to data array
        assert 2 not in dar.stream_id.values

    def test_misbalanced_array(self):
        """ test the behavior when one channel is much shorter than others """
        # make streamdict where one channel is very short
        stream_dict = {0: obspy.read(), 1: obspy.read(), 2: obspy.Stream()}
        stream_dict[0][1].data = stream_dict[0][0].data[:100]
        # get expected len and create data array
        expected_len = len(stream_dict[0][0].data)
        with pytest.warns(UserWarning) as w:
            dar = obspy_to_array(stream_dict)
        assert len(w) > 0
        assert abs(len(dar.time) - expected_len) <= 1

    def test_short_stream(self, stream_dict_short_len):
        """ test for when a waveforms is much shorter than the others """
        with pytest.warns(UserWarning) as w:
            ar = obspy_to_array(stream_dict_short_len)
        # ensure a warning was raised, some depreciation warnings creep in
        assert len(w) > 0
        # the shorter stream_dict key should have been deleted
        assert 2 not in ar.stream_id.values
        assert not ar.isnull().any()

    def test_stream_uneven(self, st_dict_close_lens):
        """ tests for when all streams are different lens """
        ar = obspy_to_array(st_dict_close_lens)
        assert not ar.isnull().any()


class TestStream2DataArray:
    """ ensure the stream2dataset function works as expected """

    # fixtures
    @pytest.fixture()
    def data_array_from_stream(self, stream):
        return obspy_to_array(stream)

    @pytest.fixture(scope="class")
    def data_array_from_gappy_stream(self, basic_stream_with_gap):
        st = basic_stream_with_gap[0]
        return obspy_to_array(st.merge(method=1), trim_stream=False)

    @pytest.fixture(scope="class")
    def data_array_from_trace_dict(self):
        """ make a data array from a dict of traces (not streams)"""
        st = pytest.waveforms["default"]
        out = {}
        for num, tr in enumerate(st):
            tr.stats.channel = st[0].stats.channel
            out[num] = tr
        return obspy_to_array(out)

    # tests
    def test_type(self, data_array_from_stream):
        """ ensure a dataset was returned """
        assert isinstance(data_array_from_stream, xr.DataArray)

    def test_shape(self, data_array_from_stream, stream):
        """ make sure the dataset matched the req_len of waveforms (plus time) """
        assert data_array_from_stream.shape[1] == len(stream)

    def test_attrs(self, data_array_from_stream):
        """ make sure the dataset has the starttime coordinate """
        attrs = data_array_from_stream.attrs
        assert {"sampling_rate"}.issubset(attrs)

    def test_stream_with_gap(self, data_array_from_gappy_stream, basic_stream_with_gap):
        """ make sure the gappy waveforms worked """
        dar = data_array_from_gappy_stream
        st, st1, st2 = basic_stream_with_gap

        starttime = np.unique(dar.starttime)[0]
        # make sure end time are the same
        ds_t2 = obspy.UTCDateTime(dar.time[-1] + starttime)
        st_t2 = max([tr.stats.endtime for tr in st])
        assert ds_t2 == st_t2
        # make sure start times are the same
        ds_t1 = obspy.UTCDateTime(starttime)
        st_t1 = min([tr.stats.starttime for tr in st])
        assert ds_t1 == st_t1
        assert dar.isnull().any()
        sr = st1[0].stats.sampling_rate
        gapsize_st = (st2[0].stats.starttime - st1[0].stats.endtime) * sr
        gapsize_ds = float(dar.isnull().sum(dim="time").values[0][0])
        # should be less than 2 sample sizes
        assert abs(gapsize_ds - gapsize_st) < 2

    def test_trace_dict_has_all_dims(self, data_array_from_trace_dict):
        """ ensure an array created from a dict of traces has all the dims """
        dims = ("stream_id", "seed_id", "time")
        assert data_array_from_trace_dict.dims == dims


class TestTrace2DataArray:
    array_fixtures = ["data_array_from_trace"]

    # fixtures
    @pytest.fixture(scope="class", params=pytest.waveforms.keys)
    def trace(self, request):
        """ return the first trace of the waveforms for testing """
        return pytest.waveforms[request.param][0]

    @pytest.fixture(scope="class")
    def data_array_from_trace(self, trace):
        """ convert the trace to a data array, return data array """
        return obspy_to_array(trace), trace

    # tests
    def test_type(self, data_array_from_trace):
        """ ensure correct type returned """
        assert isinstance(data_array_from_trace[0], xr.DataArray)

    def test_time_dimension(self, data_array_from_trace):
        """ ensure the time dimension is correct """
        array, trace = data_array_from_trace
        time = array.time
        t1 = array.starttime.values.flatten()[0]
        assert float(time.min() + t1) == trace.stats.starttime.timestamp
        assert float(time.max() + t1) == trace.stats.endtime.timestamp
        assert (array.values == trace.data).all()

    def test_stats_as_metadata(self, data_array_from_trace):
        """ ensure the stats dict was copied to attrs """
        array, trace = data_array_from_trace
        assert hasattr(array, "attrs")
        stream_id = array.stream_id.values[0]
        seed_id = array.seed_id.values[0]
        ar_stats = array.attrs["stats"][stream_id][seed_id]
        tr_stats = trace.stats
        tr_stats.pop("processing", None)
        ar_stats.pop("processing", None)
        assert ar_stats == tr_stats


class TestArray2Dict:
    """ tests for converting DataArrays back to obspy objects """

    number_of_streams = 10

    def _remove_processing(self, st):
        """ copy stream and remove processing"""
        st = st.copy()
        for tr in st:
            tr.stats.pop("processing", None)
        return st

    def equal_without_processing(self, st1, st2):
        """ Return True if the streams are equal when processing in stats is
        removed """
        st1, st2 = self._remove_processing(st1), self._remove_processing(st2)
        # make sure lengths are the same
        for tr1, tr2 in zip(st1, st2):
            min_len = min(len(tr1.data), len(tr2.data))
            tr1.data = tr1.data[:min_len]
            tr2.data = tr2.data[:min_len]
        return st1 == st2

    # fixtures
    @pytest.fixture(scope="class")
    def default_array2dict(self, default_array):
        """ convert the default array to a dict of obspy streams """
        return obsplus.array_to_obspy(default_array)

    @pytest.fixture(scope="class")
    def long_stream_dict(self):
        """ return a long dict of the default waveforms """
        st = obspy.read()
        stream_dict = {
            f"event_{item}": st.copy() for item in range(self.number_of_streams)
        }
        return stream_dict

    @pytest.fixture(scope="class")
    def array_long_stream_dict(self, long_stream_dict):
        """ convert the long waveforms dict to an array """
        return obsplus.array_to_obspy(long_stream_dict)

    # tests
    def test_stream_ids_are_keys(self, default_array, default_array2dict):
        """ ensure the default array waveforms ids made it into the dict """
        stream_ids = set(default_array.stream_id.values)
        assert stream_ids == set(default_array2dict)

    def test_streams_equal(self, default_array2dict, default_array):
        """ ensure the streams before and after array transform are equal """
        st1 = pytest.waveforms["default"].sort()
        da = default_array2dict
        st2 = da[0].sort()
        # ensure data are the same
        assert self.equal_without_processing(st1, st2)

    def test_crandall_arrays(self):
        """ Ensure the crandall canyon arrays can be converted back """
        # create data arrays from crandall
        ds = obsplus.load_dataset("crandall")
        fetcher = ds.get_fetcher()
        st_dict = dict(fetcher.yield_event_waveforms(10, 50))
        dars = list(obsplus.obspy_to_array_dict(st_dict).values())
        assert len(dars), "only expected two sampling rates"
        # convert each data array back to a stream dict
        st_dict1 = dars[0].ops.to_stream()
        st_dict2 = dars[1].ops.to_stream()
        # create stream dicts and recombine data array dicts
        for stream_id in set(st_dict1) & set(st_dict2):
            st1 = st_dict1[stream_id] + st_dict2[stream_id]
            st2 = st_dict[stream_id]
            if not self.equal_without_processing(st1, st2):
                self.equal_without_processing(st1, st2)


class TestList2Array:
    """ tests for converting a list of waveforms to a dataarray. """

    list1_len = 4

    @pytest.fixture
    def list_array1(self):
        """ A  list of homogeneous streams """
        return obspy_to_array([obspy.read() for _ in range(self.list1_len)])

    @pytest.fixture
    def list_array2(self):
        """ A list of heterogeneous traces/streams """
        # change the channel names on the default trace
        st_odd = obspy.read()
        for tr in st_odd:
            tr.stats.station = "TMU"
        # create a list and return data array
        wavelist = [obspy.read(), obspy.read()[1], obspy.read()[0], st_odd]
        return obspy_to_array(wavelist)

    @pytest.fixture(params=["list_array1", "list_array2"])
    def list_array(self, request):
        return request.getfixturevalue(request.param)

    def test_dims(self, list_array):
        stream_ids = list_array.stream_id.values
        assert len(stream_ids) == len(list_array)
        # ints beginning at one should be used as values
        assert list(stream_ids) == list(range(4))


class TestWriteDataArray:
    """ ensure the data array can be written to disk """

    # fixtures
    @pytest.fixture(scope="class")
    def written_array(self, default_array):
        """ write the default array to disk, return path """

        # TODO finish this


class TestAttachPicks:
    # tests
    def test_ptime_stime_coords(self, dar_attached_picks):
        """ ensure the coordinates ptime stime exist """
        assert "origin_time" in dar_attached_picks.coords
        assert "p_time" in dar_attached_picks.coords
        assert "s_time" in dar_attached_picks.coords

    def test_some_valid_values(self, dar_attached_picks):
        """ ensure at least some P values made it to the coords """
        vals = dar_attached_picks.coords["p_time"]
        assert (~pd.isnull(vals)).any()

    def test_dtypes(self, dar_attached_picks):
        """ the datatype of the picks columns should be float """
        for phase in ["p_time", "s_time"]:
            col = getattr(dar_attached_picks, phase)
            dtype = col.dtype
            assert not dtype.hasobject  # shouldn't be an object dtype

    def test_starttime_origin_time_seperation(self, dar_attached_picks):
        """ ensure the start of the trace and start of the events arent too
        far off """
        dar = dar_attached_picks
        cat = dar.attrs["events"]
        for ev in cat:
            rid = ev.resource_id
            time = ev.origins[-1].time.timestamp
            dd = dar[dar.stream_id == rid]
            assert (dd.origin_time.values - time == 0).all()
            assert ((dd.starttime.values - time) < 100).all()

    def test_starttime_pick_separation(self, dar_attached_picks):
        """ ensure the picks are close to starttime """
        dar = dar_attached_picks
        assert starttimes_consistent(dar, tolerance=0.02)
        picks = dar.p_time
        starttime = dar.starttime
        diff = picks - starttime
        assert (diff.isnull().values | (abs(diff) < 100).values).all()

    def test_attach_picks_many_events(self, crandall_data_array):
        """ tests for attaching picks with many (>1) events """
        p_picks = crandall_data_array.coords["p_time"]
        # all events should have at least one p pick
        all_null_events = p_picks.isnull().all(dim="seed_id")
        assert not all_null_events.any()


class TestTrimArray:
    """ tests for waveform channel trimming """

    time2add = 1
    common_fixtures = [
        "trimmed_dar",
        "trimmed_dar_ref_starttime",
        "trimmed_group",
        "trimmed_on_stream_id",
    ]

    # fixtures
    @pytest.fixture(scope="class")
    def trimmed_dar(self, dar_attached_picks):
        """ copy the data array with attached picks and trim the channels
        independently """
        dar = copy.deepcopy(dar_attached_picks)
        return dar.ops.trim(trim="p_time")

    @pytest.fixture(scope="class")
    def dar_ref_starttime(self, default_array):
        """ data array with coord that reference start of trace """
        dar = copy.deepcopy(default_array)
        add = np.ones(dar.shape[:-1]) * self.time2add
        dar.coords["something"] = (("stream_id", "seed_id"), add)
        return dar

    @pytest.fixture(scope="class")
    def trimmed_dar_ref_starttime(self, dar_ref_starttime):
        """ apply the trim function with coords that reference starttime """
        return dar_ref_starttime.ops.trim(trim="something", is_timestamp=False)

    @pytest.fixture(scope="class")
    def attached_picks_and_groups(self, dar_attached_picks):
        """ apply groups to the attached_picks """
        dar = dar_attached_picks.copy(deep=True)
        # make two groups for testing
        ones = np.ones_like(dar.starttime)
        ravel = ones.ravel()
        ravel[0 : len(ravel) // 2] = 2
        dar.coords["group"] = (dar.starttime.dims, ravel.reshape(ones.shape))
        return dar

    @pytest.fixture(scope="class")
    def trimmed_group(self, attached_picks_and_groups):
        """ apply the trim groups to the data array """
        dar = attached_picks_and_groups.copy(deep=True)
        return dar.ops.trim(trim="p_time", aggregate_by="group")

    @pytest.fixture(scope="class")
    def trimmed_on_stream_id(self, dar_attached_picks):
        """ return trimmed results on a coord that is set on stream_id """
        dar = dar_attached_picks.copy(deep=True)
        time2add = self.time2add * np.ones(np.shape(dar.stream_id))
        ser = pd.Series(time2add, index=dar.stream_id.values)
        ser.index.name = "stream_id"
        dar.coords["trimin"] = ser
        dar.ops.trim(trim="trimin", is_timestamp=False)
        return dar.ops.trim(trim="trimin", is_timestamp=False)

    @pytest.fixture(scope="class", params=common_fixtures)
    def all_trimmed_dar(self, request):
        """ metafixture for gathering all trimmed results for common tests """
        return request.getfixturevalue(request.param)

    @pytest.fixture(scope="class")
    def crandall_trimed_on_p_time(self, crandall_data_array):
        # TODO start here
        return crandall_data_array.ops.trim("p_time", aggregate_by="station")

    # specific tests
    def test_no_modification_to_original(self, dar_attached_picks):
        """ ensure there is no modification to the original data array """
        dar1 = dar_attached_picks.copy(deep=True)
        assert dar1.equals(dar_attached_picks)
        _ = dar_attached_picks.ops.trim(trim="p_time")
        assert dar1.equals(dar_attached_picks)

    def test_starttimes_are_ptimes(self, trimmed_dar):
        """ the start times should now coincide with the p_times """
        ptimes = trimmed_dar.coords["p_time"]
        starttimes = trimmed_dar.coords["starttime"]
        # either there should be no p_pick time or it should be nearly
        # equal to the start time (because trimming has already happened)
        assert ((ptimes - starttimes <= 0.01) | pd.isnull(ptimes)).all()

    def test_relative_trim(self, trimmed_dar_ref_starttime, default_array):
        """ ensure the relative start time stuff works """
        dar = trimmed_dar_ref_starttime
        expected_times = default_array.starttime + self.time2add
        assert (dar.starttime == expected_times).all()

    def test_attrs_kept(self, trimmed_dar):
        """ ensure the attrs propagated """
        assert hasattr(trimmed_dar, "attrs")

    def test_all_times_changed(self, trimmed_group, attached_picks_and_groups):
        """ since each waveform was in a group that had a change, all of the
        start times should now be different
        """
        dar1, dar2 = trimmed_group, attached_picks_and_groups
        group_has_pick = ~dar1.p_time.groupby("group").max().isnull().to_pandas()
        groups_with_no_picks = group_has_pick[~group_has_pick]
        startimes_not_equal = ~(dar1.starttime == dar2.starttime)
        in_no_pick_group = dar1.starttime.group.isin(groups_with_no_picks.index)
        # either the starttime should have moved or the starttime should belong
        # to a group that did not have any picks
        assert (startimes_not_equal | in_no_pick_group).all()

    def test_trim_stream_id_passed(self, trimmed_on_stream_id, dar_attached_picks):
        """ ensure the starttimes were adjusted """
        dar1, dar2 = trimmed_on_stream_id, dar_attached_picks
        assert (dar1.starttime != dar2.starttime).all()

    def test_trim_constant(self, default_array):
        """ test that passing a constant trims the array """
        old_start = default_array.starttime
        out = default_array.ops.trim(10, is_timestamp=False)
        new_start = out.starttime
        assert np.all((new_start - old_start) == 10)

    def test_trim_dataarray(self, default_array):
        """ test that passing a dataarray trims the array """
        old_start = default_array.starttime
        out = default_array.ops.trim(old_start + 10)
        new_start = out.starttime
        assert np.all((new_start - old_start) == 10)

    def test_trim_on_p_time(self, crandall_trimed_on_p_time):
        """ ensure the pick is about in the same sample spacing. """
        dar = crandall_trimed_on_p_time

    # common tests
    def test_starttimes_intacted(self, all_trimmed_dar):
        """ the starttimes should all be floats (non-nan) """
        assert (~pd.isnull(all_trimmed_dar.starttime)).all()

    def test_all_times_unique(self, all_trimmed_dar):
        """ ensure the time vector contains only unique values """
        flat_times = np.ravel(all_trimmed_dar.time.values)
        assert len(np.unique(flat_times)) == len(flat_times)

    def test_no_nan_values_near_start(self, all_trimmed_dar):
        """ ensure all the values at the start of the trace are not NaN """
        tmin, tmax = all_trimmed_dar.time.min(), all_trimmed_dar.time.max()
        t1 = tmin
        t2 = tmin + (tmax - tmin) / 2.0
        non_nans = all_trimmed_dar.sel(time=slice(float(t1), float(t2)))
        assert np.all(~pd.isnull(non_nans.values))

    def test_no_private_trim_coord(self, all_trimmed_dar):
        """ ensure the _trim coord has been removed """
        assert "_trim" not in all_trimmed_dar.coords


# ------------------------- test waveforms and trace functions


class TestPadTime:
    total_time = 60
    time_before = 15
    time_after = 15
    negative_time = -10

    # fixtures
    @pytest.fixture(scope="class")
    def padded_array(self, default_array):
        """ pad the default array """
        return pad_time(default_array, total_time=self.total_time)

    # tests

    def test_type(self, padded_array):
        """ tests to run on the padded array """
        assert isinstance(padded_array, xr.DataArray)

    def test_attrs_survived(self, padded_array):
        """ make sure the attrs dict came through """
        assert padded_array.attrs

    def test_pad_time(self, padded_array):
        """ tests to run on the padded array """
        pads = padded_array.sel(time=slice(30, 60)).values
        np.allclose(pads, np.zeros_like(pads))
        sr = 1.0 / padded_array.attrs["sampling_rate"]
        times = padded_array.coords["time"].values
        assert abs(times[-1] - self.total_time) <= 2 * sr

    def test_start_end_padding(self, default_array):
        """ ensure the end and start can be padded """
        padded = pad_time(
            default_array, time_after=self.time_after, time_before=self.time_before
        )

        dtimes = default_array.time
        times = padded.time
        assert times[0] == -self.time_before
        duration = times[-1] - times[0]
        dduration = dtimes[-1] - times[0]
        assert duration > dduration

    def test_start_zeroed(self, default_array):
        """ ensure the zeroed array starts at zero """
        zeroed = pad_time(
            default_array, time_before=self.time_before, start_at_zero=True
        )

        t1 = zeroed.time
        t2 = default_array.time
        assert t1[0] == 0
        assert t1[-1] > t2[-1]

    def test_trimmed_array(self, default_array):
        """ ensure the array was trimmed when a negative time_before is used
        """
        negative_pad_time = pad_time(default_array, time_before=self.negative_time)

        old_time = default_array.time.values
        new_time = negative_pad_time.time.values
        assert old_time[0] - new_time[0] == self.negative_time

    def test_trimmed_array_zero_start(self, default_array):
        """ ensure when start at 0 is used the start times get updated """
        pad = default_array.ops.pad(time_before=self.negative_time, start_at_zero=True)

        # ensure the start of the time vectors are the same
        old_time = default_array.time.values
        new_time = pad.time.values
        assert old_time[0] == new_time[0]
        assert np.isclose(new_time[-1] - old_time[-1], self.negative_time)
        # make sure starttimes were updated
        old_starttime = default_array.starttime
        new_starttime = pad.starttime
        assert (old_starttime - new_starttime == self.negative_time).all()


class TestBinArray:
    """ tests for binning values in an array """

    # fixtures
    @pytest.fixture(scope="class")
    def default_bins(self, default_array):
        """ get limit bins for the default array """
        return np.linspace(default_array.min(), default_array.max(), 100)

    @pytest.fixture(scope="class")
    def binned_default_array(self, default_array, default_bins):
        return bin_array(default_array, bins=default_bins)

    # tests
    def test_dims(self, binned_default_array, default_array, default_bins):
        """ ensure the shape and dimension labels are correct """
        dar = binned_default_array
        assert dar.dims[:-1] == default_array.dims[:-1]
        assert np.all(dar.bin == default_bins[:-1])
        assert np.all(dar.coords["upper_bin"] == default_bins[1:])

    def test_bad_bins_raise(self, default_array, default_bins):
        """ ensure bad bins raise with default raise_on_limit """
        new_dar = default_array.copy(deep=True) * 1000
        with pytest.raises(ValueError) as exec_info:
            bin_array(new_dar, default_bins)
        assert "out of bounds" in str(exec_info.value)


class TestAggregations:
    common_fixtures = ["zoo_mean_station", "zoo_max_network", "zoo_std_all"]
    seed_cols = ["network", "station", "location", "channel"]

    # helper functions
    def split_seeds(self, dar):
        """ split the seed_ids, return a df with columns: network, station,
        location, channel, if all are present, else return a subset """
        ser = dar.seed_id.to_dataframe()["seed_id"]
        split = ser.str.split(".", expand=True)
        # reset columns to seed_col names
        split.columns = self.seed_cols[: max(split.columns) + 1]
        return split.reset_index(drop=True)

    @pytest.fixture
    def zoo_mean_station(self, dar_disjoint_seeds):
        """ aggregate with mean on the station level """
        return aggregate(dar_disjoint_seeds, "mean", "station")

    @pytest.fixture
    def zoo_max_network(self, dar_disjoint_seeds):
        """ aggregate with max on the network level """
        return aggregate(dar_disjoint_seeds, "max", "network")

    @pytest.fixture
    def zoo_std_all(self, dar_disjoint_seeds):
        """ aggregate with max on the network level """
        return aggregate(dar_disjoint_seeds, "std", "all")

    @pytest.fixture(params=common_fixtures)
    def aggregated(self, request):
        """ parametrize all aggregation fixtures to run through basic common
        tests """
        return request.getfixturevalue(request.param)

    @pytest.fixture
    def aggregated_with_group_1d(self, dar_disjoint_seeds):
        """ Test aggregation with a dependent coordinate along one dim """
        # swap out time series
        dar = dar_disjoint_seeds.copy(deep=True)
        dar.values = rand.rand(*dar.values.shape)
        # add group coord
        dar.coords["group"] = "bob_" + dar.seed_id
        return aggregate(dar, method="mean", level="station", coord="group")

    @pytest.fixture
    def aggregated_with_group_2d(self, dar_disjoint_seeds):
        """ Test aggregation with a dependent coordinate along one dim """
        # swap out time series
        dar = dar_disjoint_seeds.copy(deep=True)
        dar.values = rand.rand(*dar.values.shape)
        # create groups with dims of seed_id and stream_id
        df = dar.coords["starttime"].to_pandas()
        df[0] = "1" + df.index
        df[1] = "2" + df.index
        dar.coords["group"] = df
        return aggregate(dar, method="mean", level="station", coord="group")

    # general tests
    def test_type(self, aggregated):
        """ ensure a data array was returned """
        assert isinstance(aggregated, xr.DataArray)
        assert "seed_id" in aggregated.coords

    # specific tests
    def test_mean_station(self, zoo_mean_station):
        """ ensure the level=station and method=mean works """
        df = self.split_seeds(zoo_mean_station)
        assert set(df.network) == {"UU", "BW"}
        assert set(df.station) == {"BOB", "RJOB"}
        assert "channel" not in df.columns
        assert "location" not in df.columns

    def test_max_network(self, zoo_max_network):
        """ ensure the level=station and method=mean works """
        df = self.split_seeds(zoo_max_network)
        assert set(df.network) == {"UU", "BW"}
        assert "station" not in df.columns
        assert "channel" not in df.columns
        assert "location" not in df.columns

    def test_std_all(self, zoo_std_all):
        """ ensure the all method returns only 1 seed_id """
        assert zoo_std_all.to_dataset(name="var").dims["seed_id"] == 1

    def test_groups_are_truncated_1d(self, aggregated_with_group_1d):
        """ ensure the group coord is truncated to reflect the station
        aggregation """
        ar = aggregated_with_group_1d
        groups = ar.coords["group"]
        for group in groups.values.flatten():
            assert len(str(group).split(".")) == 2  # only station level

    def test_groups_are_truncated_2d(self, aggregated_with_group_2d):
        """ ensure the group coord is truncated to reflect the station
        aggregation """
        ar = aggregated_with_group_2d
        groups = ar.coords["group"]
        for group in groups.values.flatten():
            assert len(str(group).split(".")) == 2  # only station level

    def test_aggregate_with_ufunc(self, data_array_from_dict):
        """ Ensure aggregations can be performed with universal functions. """
        dar = data_array_from_dict
        out1 = dar.ops.agg("mean", "station")
        out2 = dar.ops.agg(np.mean, "station")
        assert (out1 == out2).all()

    def test_aggregate_non_xarray_function(self, data_array_from_dict):
        """ functions that dont have xarray methods should also work. """
        dar = data_array_from_dict
        out = dar.ops.agg(np.linalg.norm, "station")
        assert isinstance(out, xr.DataArray)


class TestFFT3D:
    """ test for fft on higher dimensional sets """

    # fixtures
    @pytest.fixture(scope="class")
    def fft_3d(self, data_array_from_dict):
        """" run the 3d array through the fft """
        return array_rfft(data_array_from_dict)

    # tests
    def test_type(self, fft_3d):
        """ make sure an array was returned """
        assert isinstance(fft_3d, xr.DataArray)

    def test_dims(self, fft_3d, data_array_from_dict):
        """ make sure the dim order is the same, swapping freq for time """
        assert fft_3d.dims[:-1] == data_array_from_dict.dims[:-1]
        assert fft_3d.frequency.values.any()


class TestArrayFFTAndIFFT:
    """ tests for performing fft and ifft on the detex data arrays """

    fixtures = ["default_fft", "fft_no_attrs", "default_fft_padded"]
    fft_input_names = ["default_array", "default_array_no_attrs"]
    pad_length = 30000

    # helper functions
    def arrays_are_about_equal(self, ar1, ar2):
        """ test that the data arrays are equal """
        assert ar1.shape == ar2.shape
        assert np.allclose(ar1.values, ar2.values)
        assert set(ar1.dims) == set(ar2.dims)
        # test coords in fft_input has attrs
        if ar1.attrs:
            assert ar1.attrs == ar2.attrs
            for dim in ar1.dims:
                c1, c2 = ar1[dim], ar2[dim]
                try:
                    assert np.allclose(c1.values, c2.values)
                except TypeError:
                    assert (c1 == c2).all()

    # fixtures
    @pytest.fixture(scope="class", params=fft_input_names)
    def fft_input(self, request):
        """ return the inputs for fft function """
        return request.getfixturevalue(request.param)

    @pytest.fixture(scope="class")
    def default_fft(self, default_array):
        return array_rfft(default_array)

    @pytest.fixture(scope="class")
    def default_ifft(self, default_fft):
        return array_irfft(default_fft)

    @pytest.fixture(scope="class", params=fixtures)
    def array_fft_outputs(self, fft_input):
        """ return output of all fixtures in fixtures class attr """
        return array_rfft(fft_input)

    @pytest.fixture(scope="class")
    def array_ifft_outputs(self, array_fft_outputs):
        """ return output of all fixtures in fixtures class attr """
        return array_irfft(array_fft_outputs)

    @pytest.fixture(scope="class")
    def default_array_no_attrs(self, default_array):
        """ clear the attrs on the default array, run fft, return """
        ar = default_array.copy()
        ar.attrs = {}
        return ar

    @pytest.fixture(scope="class")
    def default_array_extra_coord(self, default_array):
        """ test if extra coord are propogated """
        ar = default_array.copy()
        ar["extra_coord"] = "why!?"
        return array_rfft(ar)

    # @pytest.fixture(scope='class')
    # def multiplexed_default_fft(self, multiplexed_default_array):
    #     """ multiplex default array, then perform fft """
    #     return array_rfft(multiplexed_default_array)

    # @pytest.fixture(scope='class')
    # def multiplexed_default_ifft(self, multiplexed_default_fft):
    #     """ multiplex default array, then perform fft """
    #     return array_irfft(multiplexed_default_fft.copy())

    @pytest.fixture(scope="class")
    def default_fft_padded(self, default_array):
        """ return the padded fft of the default array """
        ar = default_array.copy()
        return array_rfft(ar, required_len=self.pad_length)

    # tests
    def test_type(self, array_fft_outputs):
        """ ensure an array was returned with complex dtype """
        assert isinstance(array_fft_outputs, xr.DataArray)
        assert array_fft_outputs.values.dtype == np.complex128

    def test_fft_outputs(self, array_fft_outputs):
        """ ensure an array was returned with complex dtype """
        assert isinstance(array_fft_outputs, xr.DataArray)
        assert array_fft_outputs.values.dtype == np.complex128

    def test_requencies(self, array_fft_outputs):
        """ ensure frequency is an axis and has both positive and negative
         numbers """
        assert "frequency" in array_fft_outputs.coords
        freqs = array_fft_outputs.coords["frequency"].values
        assert freqs[0] == 0
        if "sampling_rate" in array_fft_outputs.attrs:
            niquist = float(array_fft_outputs.attrs["sampling_rate"]) / 2.0
            assert abs(freqs[-1]) == niquist

    def test_ifft(self, array_ifft_outputs, fft_input):
        """ test that converting to and from freq. domain is low-lossy """
        ar1, ar2 = fft_input, array_ifft_outputs
        self.arrays_are_about_equal(ar1, ar2)

    def test_attrs_survived_fft(self, default_fft, default_array):
        """ ensure an array was returned with complex dtype """
        assert default_fft.attrs == default_array.attrs

    def test_attrs_survived_ifft(self, default_fft, default_ifft):
        """ ensure an array was returned with complex dtype """
        assert default_fft.attrs == default_ifft.attrs

    def test_extra_coord_survives(self, default_array_extra_coord):
        """ ensure the extra coord survived """
        assert "extra_coord" in default_array_extra_coord.coords

    def test_default_values(self, default_fft):
        """ ensure performing fft directly on trace returns same result """
        st = pytest.waveforms["default"]
        for tr in st:
            # ensure the data are the same if fft performed directly along
            # expected axis
            seed_id = tr.id
            next_fast = scipy.fftpack.next_fast_len(len(tr.data))
            tfft = np.fft.rfft(tr.data, next_fast)
            ar_fft = default_fft.loc[0, seed_id, :]
            assert (tfft == ar_fft).all()
            # try transforming back, make sure array are almost equal
            t_arr = np.fft.irfft(ar_fft, next_fast)
            assert np.allclose(t_arr, tr.data)

    def test_fft_padding(self, default_fft_padded):
        """ make sure the array is correctly padded """
        assert default_fft_padded.shape[-1] == self.pad_length // 2 + 1


class TestIterSeed:
    """ tests for iteratively slicing arrays by seed_id """

    # tests
    def test_iter_station(self, many_sid_array):
        """ ensure iterstation works """
        for dar in many_sid_array.ops.iter_seed("station"):
            df = get_nslc_df(dar)
            assert len(df.station.unique()) == 1


class TestGetSid:
    """ test for slicing dataframes based on seed ids """

    default_seed_id = "BW.RJOB..EHZ"

    # fixtures
    @pytest.fixture(scope="class")
    def sliced_by_seed(self, many_sid_array):
        """ get a particular seed_id from the default array"""
        return sel_sid(many_sid_array, self.default_seed_id)

    @pytest.fixture(scope="class")
    def sliced_by_wildcard(self, many_sid_array):
        """ slice by wildcard"""
        return sel_sid(many_sid_array, "*.*.*.EHZ")

    # tests
    def test_sliced_by_one_seed(self, sliced_by_seed):
        """ ensure only the selected seed is taken """
        df = get_nslc_df(sliced_by_seed)
        assert len(df) == 1
        expected = set([self.default_seed_id])
        assert expected == set(sliced_by_seed.seed_id.values)

    def test_slice_by_wild(self, sliced_by_wildcard):
        """ ensure the wildcard sloce works """
        df = get_nslc_df(sliced_by_wildcard)
        assert set(df.channel.values) == set(["EHZ"])

    def test_network_channel_filter(self, bingham_dar):
        """ tests for filtering on network and channel using bingham data """
        filtered_dar = bingham_dar.ops.sel_sid("UU.*.*.ENZ")
        assert len(filtered_dar.seed_id)
        for seed_id in filtered_dar.seed_id.values:
            assert seed_id.endswith("ENZ")
            assert seed_id.startswith("UU")


class TestPickle:
    """ tests for the pickle functionality on data arrays """

    # fixtures
    @pytest.fixture(scope="class")
    def pickled_path(self, default_array):
        """ save an object to a temporary file """
        with tempfile.NamedTemporaryFile() as tf:
            default_array.ops.to_pickle(tf.name)
            yield tf.name
        if os.path.exists(tf.name):
            os.remove(tf.name)

    @pytest.fixture(scope="class")
    def unpickled_from_path(self, pickled_path):
        """ unpickle the pickled path """
        return read_pickle(pickled_path)

    @pytest.fixture(scope="class")
    def pickled_bytes(self, default_array):
        """ pickle to bytes """
        return default_array.ops.to_pickle()

    @pytest.fixture(scope="class")
    def unpickled_from_bytes(self, pickled_bytes):
        """ unpickle from bytes """
        return read_pickle(pickled_bytes)

    # tests
    def test_unpickled_file(self, default_array, unpickled_from_path):
        """ ensure the pickle is non-lossy """
        assert default_array.equals(unpickled_from_path)

    def test_unpickled_bytes(self, unpickled_from_bytes, default_array):
        """ ensure byte pickling is non-lossy """
        assert unpickled_from_bytes.equals(default_array)


class Test2Netcdf:
    """ Tests for writing netcdf files from data arrays """

    output_params = ["read_netcdf_from_path"]

    # fixtures
    @pytest.fixture(scope="class")
    def default_array_more(self, default_array: xr.DataArray):
        """ return a deep copy of the default array with
        an obspy events and stations attached """
        dar = default_array.copy(deep=True)
        dar.attrs["events"] = obspy.read_events()
        dar.attrs["stations"] = obspy.read_inventory()
        return dar

    @pytest.fixture(scope="class")
    def netcdf_path(self, default_array_more):
        """ save the array to a temporary netcdf file """
        with tempfile.NamedTemporaryFile(suffix=".h5") as tf:
            default_array_more.ops.to_netcdf(tf.name)
            yield tf.name
        if os.path.exists(tf.name):
            os.remove(tf.name)

    @pytest.fixture(scope="class")
    def read_netcdf_from_path(self, netcdf_path):
        """ read the netcdf file back into memory """
        return netcdf2array(netcdf_path)

    @pytest.fixture(scope="class")
    def netcdf_bytes(self, default_array_more):
        """ convert datarray to netcdf bytes """
        return default_array_more.ops.to_netcdf()

    @pytest.fixture(scope="class")
    def read_netcdf_from_bytes(self, netcdf_bytes):
        """ read the netcdf from bytes """
        return netcdf2array(netcdf_bytes)

    @pytest.fixture(scope="class", params=output_params)
    def output_dar(self, request):
        """ metafixture for collecting outputs """
        return request.getfixturevalue(request.param)

    # tests
    def test_basics(self, output_dar):
        """ ensure the correct type and attrs exists """
        assert isinstance(output_dar, xr.DataArray)
        assert hasattr(output_dar, "attrs")
        assert output_dar.attrs  # non-empty dict

    def test_not_lossy(self, output_dar, default_array_more):
        """ make sure the IO is non-lossy """
        assert output_dar.equals(default_array_more)
        cat1 = default_array_more.attrs["events"]
        cat2 = output_dar.attrs["events"]
        assert cat1 == cat2
        inv1 = default_array_more.attrs["stations"]
        inv2 = output_dar.attrs["stations"]
        assert inv1 == inv2


class TestSeedStacking:
    """ Tests for stacking waveform arrays by seed levels """

    def test_all_level_stacks(self, crandall_data_array):
        """
        Test stacking the data array on all supported levels, and that
        it can be unstacked.
        """
        # get a dataframe which splits network, station, location, and chan
        seed_df = get_nslc_df(crandall_data_array)
        # iterate each level and stack, then unstack
        for level in seed_df.columns:
            dar1 = crandall_data_array.copy()
            dar2 = dar1.ops.stack_seed(level)
            assert len(dar2[level]) == len(seed_df[level].unique())

