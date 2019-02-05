"""
tests for the data fetcher capabilities
"""
import copy
import glob
import os
import tempfile
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
import pytest
from obspy.clients.fdsn import Client
from obspy.core.event import Event, Origin

import obsplus
from obsplus import Fetcher, WaveBank, stations_to_df
from obsplus.datasets.dataloader import DataSet
from obsplus.utils import make_time_chunks, get_reference_time

WAVEFETCHERS = []


# ----------------------------- helper functions


def append_func_name(list_obj):
    """ decorator to append a function name to list_obj """

    def wrap(func):
        list_obj.append(func.__name__)
        return func

    return wrap


def trim_kem_events(fetcher: Fetcher):
    """ trim the events in the kemmerer data set to only includes times
    when continuous data is available. """
    t1 = obspy.UTCDateTime("2009-04-01").timestamp
    t2 = obspy.UTCDateTime("2009-04-04").timestamp
    cat = fetcher.event_client.get_events(starttime=t1, endtime=t2)
    return Fetcher(
        waveforms=fetcher.waveform_client,
        events=cat,
        stations=fetcher.station_client,
        stream_processor=fetcher.stream_processor,
    )


# ---------------------------- class level fixtures


@pytest.fixture(scope="session")
@append_func_name(WAVEFETCHERS)
def kem_fetcher():
    """ init a waveform fetcher passing a path to a directory as the arg """
    return obsplus.load_dataset("kemmerer").get_fetcher()


@pytest.fixture(scope="session")
@append_func_name(WAVEFETCHERS)
def ta_fetcher(ta_dataset):
    """ init a waveform fetcher using an active obspy client and the TA
    stations """
    return ta_dataset.get_fetcher()


@pytest.fixture(scope="session")
@append_func_name(WAVEFETCHERS)
def kem_fetcher_with_processor():
    """ same as kem fetcher but with a stream_processor """
    dataset = obsplus.load_dataset("kemmerer")

    def processor(st):
        """ simple processor to apply bandpass filter """
        st.filter("bandpass", freqmin=1, freqmax=10)
        # mark that the processor ran
        for tr in st:
            tr.stats["processor_ran"] = True
        return st

    wf = Fetcher(
        waveforms=dataset.waveform_client.get_waveforms(),
        events=dataset.event_client.get_events(),
        stations=dataset.station_client.get_stations(),
        stream_processor=processor,
    )
    return trim_kem_events(wf)


@pytest.fixture(scope="session")
@append_func_name(WAVEFETCHERS)
def kem_fetcher_limited():
    """ init a fetcher with a subset of the events """
    kemmerer_dataset = obsplus.load_dataset("kemmerer")
    # load in a subset of the full events dataframe
    event_ids = {
        "smi:local/042f78e9-6089-4ed8-8f9b-47c2189a1c75",
        "smi:local/16459ce7-69ff-4fe0-8639-a980b47498bb",
    }
    df = pd.read_csv(kemmerer_dataset.path / "catalog.csv")
    df = df[df.event_id.isin(event_ids)]
    wf = Fetcher(
        waveforms=kemmerer_dataset.waveform_client,
        events=df,
        stations=kemmerer_dataset.station_client,
    )
    return wf


# ---------------------------------- tests


class TestGeneric:
    """ generic tests for wavefetchers, mostly just to make sure they run """

    important_attrs = ["yield"]
    other_attrs = ["time_before", "time_after"]

    # fixtures
    @pytest.fixture(scope="session", params=WAVEFETCHERS)
    def wavefetcher(self, request):
        return request.getfixturevalue(request.param)

    @pytest.fixture(scope="session")
    def copied_fetcher(self, wavefetcher):
        """ init a wavefetcher from a wavefetcher, return tuple of both """
        return (wavefetcher, Fetcher(wavefetcher))

    @pytest.fixture(scope="session")
    def kem_copy(self, kem_fetcher):
        """ return a copy of the kem_fetcher """
        return kem_fetcher.copy()

    # general tests
    def test_has_attrs(self, wavefetcher):
        """ test that streams for each event can be yielded  """
        assert hasattr(wavefetcher, "yield_event_waveforms")

    def test_copied_wavefetchers_get_same_data(self, copied_fetcher):
        """ ensure the two wavefetchers are equal """
        wf1, wf2 = copied_fetcher
        inv_df = wf1.station_df
        start = inv_df.start_date.min()
        end = start + 15
        # ensure the same data is returned
        ew1 = wf1.get_waveforms(start, end)
        ew2 = wf2.get_waveforms(start, end)
        assert ew1 == ew2

    def test_init_with_banks(self, bingham_dataset):
        """ Ensure the fetcher can be init'ed with all bank inputs. """
        wbank = obsplus.WaveBank(bingham_dataset.waveform_path)
        ebank = obsplus.EventBank(bingham_dataset.event_path)
        wbank.update_index(), ebank.update_index()
        sbank = bingham_dataset.station_client
        fetcher = Fetcher(waveforms=wbank, events=ebank, stations=sbank)
        edf = fetcher.event_df
        sdf = fetcher.station_df
        for df in [edf, sdf]:
            assert isinstance(df, pd.DataFrame)
            assert not df.empty


class TestGetWaveforms:
    """ tests for getting streams from the fetcher """

    duration = 10
    generic_streams = []

    # fixtures
    @pytest.fixture(scope="session")
    @append_func_name(generic_streams)
    def kem_stream(self, kem_fetcher):
        """ using the kem fetcher, return a data waveforms returned
        by get_waveforms"""
        starttime = obspy.UTCDateTime("2009-04-01")
        return kem_fetcher.get_waveforms(starttime, starttime + self.duration)

    @pytest.fixture(scope="session")
    @append_func_name(generic_streams)
    def kem_stream_processed(self, kem_fetcher_with_processor):
        """ using the kem fetcher, return a data waveforms returned
        by get_waveforms"""
        starttime = obspy.UTCDateTime("2009-04-01")
        fetch = kem_fetcher_with_processor
        return fetch.get_waveforms(starttime, starttime + self.duration)

    @pytest.fixture(scope="session")
    @append_func_name(generic_streams)
    def ta_stream(self, ta_fetcher):
        """ using the ta_fetcher and get_waveforms, return a data waveforms """
        starttime = obspy.UTCDateTime("2007-02-20")
        try:
            return ta_fetcher.get_waveforms(starttime, starttime + self.duration)
        except obspy.clients.fdsn.header.FDSNException:
            pytest.skip("failed to communicate with IRIS")

    @pytest.fixture(scope="session", params=generic_streams)
    def stream(self, request):
        """ a meta fixture for collecting all streams to perform common test
        on"""
        return request.getfixturevalue(request.param)

    # general waveforms tests tests
    def test_streams_basics(self, stream):
        """ ensure a non-empty waveforms was returned """
        assert isinstance(stream, obspy.Stream)
        assert len(stream)

    def test_stream_durations(self, stream):
        """ ensure a waveforms of the correct length (under 2 samples)
        was returned """
        for tr in stream:
            t1 = tr.stats.starttime
            t2 = tr.stats.endtime
            duration = t2.timestamp - t1.timestamp
            diff = duration - self.duration
            assert abs(diff) < 1.5 / tr.stats.sampling_rate

    # specific tests
    def test_stream_processor_ran(self, kem_stream_processed):
        """ ensure the waveforms returned has been run through the
         custom processing function """
        for tr in kem_stream_processed:
            assert tr.stats["processor_ran"]


class TestYieldWaveforms:
    """ tests for yielding chunks of data between a time range """

    duration = 10.0
    overlap = 2.0
    starttime = obspy.UTCDateTime("2009-04-01")
    endtime = starttime + 60.0

    # helper functions
    def check_duration(self, st):
        """ ensure the durations are approximately correct """
        for tr in st:
            duration = abs(tr.stats.endtime - tr.stats.starttime)
            tolerance = 1.5 / tr.stats.sampling_rate
            if (duration - (self.duration + self.overlap)) >= tolerance:
                return False
        return True

    def check_stream_processor_ran(self, st):
        for tr in st:
            try:
                if not tr.stats["processor_ran"]:
                    return False
            except (AttributeError, KeyError):
                return False
        return True

    # fixtures
    @pytest.fixture(scope="session")
    def kem_stream(self, kem_fetcher_with_processor):
        """ return a list of streams yielded from fetcher """
        fet = kem_fetcher_with_processor
        return list(
            fet.yield_waveforms(
                self.starttime, self.endtime, self.duration, self.overlap
            )
        )

    # tests
    def test_durations(self, kem_stream):
        """ensure the duration are as expected """
        for st in kem_stream:
            assert self.check_duration(st)

    def test_stream_processor_ran(self, kem_stream):
        """ ensure the waveforms processor ran on each waveforms """
        for st in kem_stream:
            for tr in st:
                assert tr.stats["processor_ran"]


class TestYieldEventWaveforms(TestYieldWaveforms):
    """ tests for getting waveforms that correspond to events """

    timebefore = 1
    timeafter = 10
    duration = timeafter + timeafter
    commons = []

    # helper functions
    def check_stream_dict(self, st_dict):
        """ test waveforms dict """
        assert isinstance(st_dict, dict) and st_dict
        for name, st in st_dict.items():
            assert isinstance(st, obspy.Stream) and len(st)

    # fixtures
    @pytest.fixture(scope="session")
    @append_func_name(commons)
    def event_list_origin(self, kem_fetcher_with_processor):
        """ return a list of event waveforms, each starttime referenced
        at origin """
        func = kem_fetcher_with_processor.yield_event_waveforms
        return list(func(self.timeafter, self.timeafter))

    @pytest.fixture(scope="session")
    @append_func_name(commons)
    def event_list_p(self, kem_fetcher_with_processor):
        """ return a list of event waveforms, each starttime referenced
        at the pwave for the channel """
        func = kem_fetcher_with_processor.yield_event_waveforms
        return list(func(1, 10, reference="p"))

    @pytest.fixture(scope="session", params=commons)
    def stream_list(self, request):
        """ collect all waveforms lists to apply general tests on """
        return request.getfixturevalue(request.param)

    @pytest.fixture(scope="session")
    def stream_dict_zero_starttime(self, kem_fetcher):
        """ yield waveforms into a dict using 0 for starttimes """
        return dict(kem_fetcher.yield_event_waveforms(0, self.timeafter))

    @pytest.fixture(scope="class")
    def wavefetch_no_inv(self, bingham_dataset):
        """ init wavefetcher with inv_df zeroed """
        kwargs = dict(
            waveforms=bingham_dataset.waveform_client,
            events=bingham_dataset.event_client,
        )
        return Fetcher(**kwargs)

    # general test
    def test_duration(self, stream_list):
        """ ensure the duration of the streams is correct """
        for _, st in stream_list:
            assert self.check_duration(st)

    def test_stream_processor(self, stream_list):
        """ ensure the waveforms processor ran """
        assert len(stream_list)
        for _, st in stream_list:
            assert self.check_stream_processor_ran(st)

    # phase tests
    def test_only_p_phases(self, event_list_p, kem_fetcher_with_processor):
        """ make sure only stations that have p picks are returned """
        df = kem_fetcher_with_processor.picks_df
        for eve_id, st in event_list_p:
            pick_df = df[df.event_id == eve_id]
            assert len(pick_df), f"{st} has no picks!"
            assert set(pick_df.station) == {tr.stats.station for tr in st}

    # zero starttime test
    def test_zero_starttime(self, stream_dict_zero_starttime):
        """ test that zero starttimes doesnt throw an error """
        for eve_id, stream in stream_dict_zero_starttime.items():
            if not len(stream):
                continue
            assert isinstance(stream, obspy.Stream)
            sr = stream[0].stats.sampling_rate
            t1, t2 = stream[0].stats.starttime, stream[0].stats.endtime
            duration = abs(t2.timestamp - t1.timestamp)
            assert abs(duration - self.timeafter) < 2 * sr

    def test_correct_stream(self, bingham_dataset, bingham_stream_dict):
        """ ensure the correct streams are given for ids """
        cat = bingham_dataset.event_client.get_events()
        evs = {str(ev.resource_id): ev for ev in cat}
        for eve_id, st in bingham_stream_dict.items():
            ev = evs[eve_id]
            time2 = get_reference_time(ev).timestamp
            tmin = min([tr.stats.starttime.timestamp for tr in st])
            assert abs(tmin - time2) < 12

    # wavefetch with no stations
    def test_yield_event_waveforms_no_inv(self, wavefetch_no_inv):
        """ WaveFetchers backed by WaveBanks should be able to pull
        station data from wavebank index df in most cases. """
        # ensure the inv_df is not None
        inv_df = wavefetch_no_inv.station_df
        assert inv_df is not None
        assert not inv_df.empty
        kwargs = dict(time_after=10, time_before=20, reference="p")
        st_dict = dict(wavefetch_no_inv.yield_event_waveforms(**kwargs))
        self.check_stream_dict(st_dict)


class TestStreamProcessor:
    """ ensure the waveforms processors get called """

    # fixtures
    @pytest.fixture(scope="session")
    def fetcher(self, kem_fetcher):
        """ return the waveform fetcher and overwrite the stream_processor """
        new_fetcher = copy.deepcopy(kem_fetcher)

        def stream_processor(st: obspy.Stream) -> obspy.Stream:
            """ select the z component, detrend, and filter a waveforms """
            # st = st.select(component="Z")
            # st.detrend("linear")
            # st.filter("bandpass", freqmin=1, freqmax=10)
            return st

        new_fetcher.stream_processor = stream_processor

        return new_fetcher

    @pytest.fixture(scope="session")
    def stream_list(self, fetcher):
        t1 = obspy.UTCDateTime("2009-04-01T00-00-00")
        t2 = obspy.UTCDateTime("2009-04-01T03-59-59")
        duration = 7200
        overlap = 60
        return list(fetcher.yield_waveforms(t1, t2, duration, overlap))

    # tests
    def test_streams_only_z_components(self, stream_list):
        """ ensure the st.select part of the waveforms processor discarded other
        channels """
        for st in stream_list:
            for tr in st:
                assert tr.id.endswith("Z")


class TestSwapAttrs:
    """ ensure events, stations, and picks objects can be temporarily swapped out """

    t1 = obspy.UTCDateTime("2009-04-01T00-01-00")
    tb = 1
    ta = 9

    # fixtures
    @pytest.fixture(scope="session")
    def catalog(self):
        """ assemble a events to test yield_event_waveforms with an event
        that was not initiated from the start """
        ori = Origin(time=self.t1, latitude=47.1, longitude=-100.22)
        event = Event(origins=[ori])
        return obspy.Catalog(events=[event])

    @pytest.fixture(scope="session")
    def new_event_stream(self, kem_fetcher, catalog):
        """ get a single event from the fetcher, overwriting the attached
        events """
        func = kem_fetcher.yield_event_waveforms
        result = func(time_before=self.tb, time_after=self.ta, events=catalog)
        return list(result)[0].stream

    @pytest.fixture(scope="session")
    def yield_event_streams(self, kem_fetcher, kem_archive):
        """ yield a subset of the events in kem_fetcher """
        path = Path(kem_archive).parent / "catalog.csv"
        event_df = pd.read_csv(path).iloc[0:1]
        tb = 1
        ta = 3
        ite = kem_fetcher.yield_event_waveforms(tb, ta, events=event_df)
        return list(ite)

    @pytest.fixture(scope="class")
    def new_inventory_df(self, kem_fetcher):
        """ return a new stations dataframe with only the first row """
        return kem_fetcher.station_df.iloc[0:1]

    @pytest.fixture(scope="class")
    def new_inventory_stream(self, kem_fetcher, new_inventory_df):
        """ try swapping out the stations to only retutn a subset of the
        channels """
        t1 = self.t1
        t2 = self.t1 + 60
        return kem_fetcher.get_waveforms(
            starttime=t1, endtime=t2, stations=new_inventory_df
        )

    # tests for events swaps
    def test_time(self, new_event_stream):
        """ ensure the new time was returned """
        assert len(new_event_stream)
        t1 = new_event_stream[0].stats.starttime.timestamp
        t2 = new_event_stream[0].stats.endtime.timestamp
        assert t1 < self.t1.timestamp < t2

    def test_iter(self, yield_event_streams):
        """ ensure the yield events worked """
        for event_id, stream in yield_event_streams:
            assert isinstance(stream, obspy.Stream)

    # tests for stations swaps
    def test_streams(self, new_inventory_stream, new_inventory_df):
        """ ensure only the cahnnel in the new_inventory df was returned """
        assert len(new_inventory_stream) == 1
        assert new_inventory_stream[0].id == new_inventory_df["seed_id"].iloc[0]


class TestCallWaveFetcher:
    """ test that calling the wavefetcher provides a simplified interface for
    getting waveforms """

    t1 = obspy.UTCDateTime("2009-04-01")
    tb = 1
    ta = 9
    duration = tb + ta

    # fixtures
    @pytest.fixture(scope="class")
    def stream(self, kem_fetcher):
        """ return a waveforms from calling the fetcher """
        return kem_fetcher(self.t1, time_before=self.tb, time_after=self.ta)

    # tests
    def test_callable(self, kem_fetcher):
        """ ensure the fetcher is callable """
        assert callable(kem_fetcher)

    def test_is_stream(self, stream):
        """ ensure the waveforms is an instance of waveforms """
        assert isinstance(stream, obspy.Stream)

    def test_channels(self, stream, kem_fetcher):
        """ ensure all channels are present """
        stream_channels = {tr.id for tr in stream}
        sta_channels = set(kem_fetcher.station_df.seed_id)
        assert stream_channels == sta_channels

    def test_stream_duration(self, stream):
        """ ensure the waveforms is of proper duration """
        stats = stream[0].stats
        duration = stats.endtime - stats.starttime
        sr = stats.sampling_rate
        assert abs(duration - self.duration) < 2 * sr

    def test_zero_in_time_before(self, kem_fetcher):
        """ ensure setting time_before parameter to 0 doesn't raise """
        starttime = obspy.UTCDateTime("2009-04-01")
        try:
            kem_fetcher(starttime, time_before=0, time_after=1)
        except AssertionError:
            pytest.fail("should not raise")

    def test_zero_in_time_after(self, kem_fetcher):
        """ ensure setting time_after parameter to 0 doesn't raise """
        starttime = obspy.UTCDateTime("2009-04-01")
        try:
            kem_fetcher(starttime, time_before=1, time_after=0)
        except AssertionError:
            pytest.fail("should not raise")


class TestYieldCallables:
    """ tests for yielding callables that return streams """

    starttime = obspy.UTCDateTime("2009-04-01")
    endtime = obspy.UTCDateTime("2009-04-01T00-00-01")
    duration = 600
    overlap = 10

    # f

    # fixtures
    @pytest.fixture(scope="session")
    def callables(self, kem_fetcher):
        it = kem_fetcher.yield_waveform_callable(
            self.starttime, self.endtime, self.duration, self.overlap
        )
        return list(it)

    @pytest.fixture(scope="session")
    def streams(self, callables):
        return [x() for x in callables]

    @pytest.fixture(scope="session")
    def time_tuple(self):
        tc = make_time_chunks(self.starttime, self.endtime, self.duration, self.overlap)
        return list(tc)

    # tests
    def test_streams(self, streams):
        """ ensure waveforms were yielded """
        for stream in streams:
            assert isinstance(stream, obspy.Stream)

    def test_times(self, streams, time_tuple):
        """ ensure the times in the yielded wavefroms are as expected"""
        for st, (t1, t2) in zip(streams, time_tuple):
            sr = st[0].stats.sampling_rate
            tt1, tt2 = st[0].stats.starttime, st[0].stats.endtime
            assert abs(t1 - tt1) < 2 * sr
            assert abs(t2 - tt2) < 2 * sr


class TestClientNoGetBulkWaveForms:
    """ test that clients without get bulk waveforms, ie earthworm, work """

    starttime = obspy.UTCDateTime("2009-04-01")
    endtime = obspy.UTCDateTime("2009-04-01T00-00-01")
    duration = 600
    overlap = 10

    # fixtures
    @pytest.fixture
    def kem_bank_no_bulk(self, kem_archive, monkeypatch):
        """ remove the get_waveforms_bulk from Sbank class """
        monkeypatch.delattr(WaveBank, "get_waveforms_bulk")
        monkeypatch.delattr(WaveBank, "get_waveforms_by_seed")
        # return a bank
        yield WaveBank(kem_archive)

    @pytest.fixture
    def wavefetcher_no_bulk(self, kem_bank_no_bulk, kemmerer_dataset):
        """ return a wavefetcher from the bank """
        inv = kemmerer_dataset.station_client.get_stations()
        return Fetcher(waveforms=kem_bank_no_bulk, stations=inv)

    @pytest.fixture
    def yielded_streams(self, wavefetcher_no_bulk):
        """ yield streams from wavefetcher """
        fun = wavefetcher_no_bulk.yield_waveforms
        ite = fun(self.starttime, self.endtime, self.duration, self.overlap)
        return list(ite)

    # tests
    def test_streams_yielded(self, yielded_streams):
        """ assert streams were yielded, ensuring get_waveforms rather
        than get_waveform_bulk was used """
        for st in yielded_streams:
            assert isinstance(st, obspy.Stream)


class TestFilterInventoryByAvailability:
    """ ensure that only times in the stations get used in get_bulk stuff """

    t0 = obspy.UTCDateTime("2015-12-01")
    t1 = obspy.UTCDateTime("2016-01-01")
    t2 = obspy.UTCDateTime("2016-02-01")

    # fixtures
    @pytest.fixture
    def altered_inv(self):
        """ return an stations with one enddate changed to a later date """
        df = stations_to_df(obspy.read_inventory())
        df.loc[:, "start_date"] = self.t0
        df.loc[:, "end_date"] = self.t1
        df.loc[0, "end_date"] = self.t2
        return df

    @pytest.fixture
    def inv_with_none(self):
        """ return an stations with one enddate changed to None """
        df = stations_to_df(obspy.read_inventory())
        df.loc[:, "start_date"] = self.t0
        df.loc[:, "end_date"] = self.t1
        df.loc[0, "end_date"] = None
        return df

    @pytest.fixture
    def bulk_arg_later_time(self, altered_inv):
        fetcher = Fetcher(None, stations=altered_inv)
        return fetcher._get_bulk_arg(self.t1 + 10, self.t2)

    @pytest.fixture
    def bulk_arg_none_enddate(self, inv_with_none):
        """ return the bulk args from an inv with None endate """
        fetcher = Fetcher(None, stations=inv_with_none)
        return fetcher._get_bulk_arg(self.t0, self.t1)

    @pytest.fixture
    def fetcher(self, altered_inv, kem_fetcher):
        """ return a fetcher with the modified times """
        return Fetcher(kem_fetcher.waveform_client, stations=altered_inv)

    # tests
    def test_bulk_arg_is_limited(self, bulk_arg_later_time, altered_inv):
        """ ensure bulk arg doesn't include times the stations doesnt
        have data """
        assert len(bulk_arg_later_time) == 1
        ba = bulk_arg_later_time[0]
        ser = altered_inv.iloc[0]
        assert ba[0] == ser.network
        assert ba[1] == ser.station
        assert ba[3] == ser.channel

    def test_none_endtimes_are_used(self, bulk_arg_none_enddate, inv_with_none):
        """ ensure any channels with enddates of None are not filtered out """
        assert len(bulk_arg_none_enddate) == len(inv_with_none)

    def test_empty_stream_from_before_start(self, fetcher):
        """ ensure when data is requested before stations starttime that an
        empty string is returned """
        st = fetcher(obspy.UTCDateTime("1970-01-01"), 10, 40)
        assert isinstance(st, obspy.Stream)
        assert not len(st)


class TestGetEventData:
    t1 = obspy.UTCDateTime("2009-04-01").timestamp
    t2 = obspy.UTCDateTime("2009-04-04").timestamp

    path = "eventwaveforms/{year}/{julday}"

    # fixtures
    @pytest.fixture(scope="class")
    def temp_dir_path(self):
        """ return a path to a temporary directory """
        with tempfile.TemporaryDirectory() as tempdir:
            out = os.path.join(tempdir, "temp")
            yield out

    @pytest.fixture(scope="class")
    def fetcher(self, kem_fetcher):
        """ make a copy of the kem_fetcher and restrict scope of events to
        when data are available."""
        fet = kem_fetcher.copy()
        df = fet.event_df
        fet.event_df = df[(df.time >= self.t1) & (df.time <= self.t2)]
        return fet

    @pytest.fixture(scope="class")
    def download_data(self, temp_dir_path, fetcher: Fetcher):
        """ download data from the kem fetcher into the tempdir, return
        path to tempdir """
        path = os.path.join(temp_dir_path, self.path)
        params = dict(time_before_origin=0, time_after_origin=10, path=path)
        breakpoint()
        fetcher.download_event_waveforms(**params)
        return temp_dir_path

    @pytest.fixture(scope="class")
    def event_sbank(self, download_data):
        """ return an sbank pointed at the temp_dir_path """
        sb = WaveBank(download_data)
        sb.update_index()
        return sb

    @pytest.fixture(scope="class")
    def event_fetcher(self, event_sbank, fetcher):
        """ init a fetcher using the old fetcher """
        fet = fetcher.copy()
        fet._download_client = event_sbank
        return fet

    @pytest.fixture(scope="class")
    def mseeds(self, download_data):
        """ return a list of all the files with the ext mseed """
        return glob.glob(os.path.join(download_data, "**", "*mseed"), recursive=True)

    @pytest.fixture(scope="class")
    def stream_dict(self, event_fetcher):
        """ return a dict of the events contained in the waveforms """
        return dict(event_fetcher.yield_event_waveforms(0, 10))

    # tests
    def test_directory_exists(self, download_data):
        """ ensure the directory was created """
        assert os.path.exists(download_data)

    def test_mseeds(self, mseeds):
        """ ensure some files with mseed ext were created """
        assert len(mseeds)

    def test_data_were_downloaded(self, stream_dict):
        """ ensure data from the events exists """
        for eveid, stream in stream_dict.items():
            assert isinstance(stream, obspy.Stream)
            assert len(stream)


class TestGetContinuousData:
    t1 = obspy.UTCDateTime("2009-04-01").timestamp
    t2 = obspy.UTCDateTime("2009-04-01T03-00-00").timestamp
    duration = 1800
    overlap = 60
    path = "continuouswaveforms/{year}/{julday}/{network}/{station}"

    # fixtures
    @pytest.fixture(scope="class")
    def temp_dir_path(self):
        """ return a path to a temporary directory """
        with tempfile.TemporaryDirectory() as tempdir:
            out = os.path.join(tempdir, "temp")
            yield out

    @pytest.fixture(scope="class")
    def download_data(self, temp_dir_path, kem_fetcher: Fetcher):
        """ download data from the kem fetcher into the tempdir, return
        path to tempdir """
        path = Path(temp_dir_path) / self.path
        params = dict(
            starttime=self.t1,
            endtime=self.t2,
            duration=self.duration,
            overlap=self.overlap,
            path=path,
        )
        kem_fetcher.download_waveforms(**params)
        return temp_dir_path

    @pytest.fixture(scope="class")
    def continuous_sbank(self, download_data):
        """ return an sbank pointed at the temp_dir_path """
        sb = WaveBank(download_data)
        sb.update_index()
        return sb

    @pytest.fixture(scope="class")
    def continuous_fetcher(self, continuous_sbank, kem_fetcher):
        """ init a fetcher using the old fetcher """
        fet = kem_fetcher.copy()
        fet._download_client = continuous_sbank
        return fet

    @pytest.fixture(scope="class")
    def mseeds(self, download_data):
        """ return a list of all the files with the ext mseed """
        return glob.glob(os.path.join(download_data, "**", "*mseed"), recursive=True)

    @pytest.fixture(scope="class")
    def stream_list(self, continuous_fetcher: Fetcher):
        """ return a dict of the events contained in the waveforms """
        utc1 = self.t1
        utc2 = self.t2
        input_dict = dict(
            starttime=utc1, endtime=utc2, duration=self.duration, overlap=self.overlap
        )
        return list(continuous_fetcher.yield_waveforms(**input_dict))

    # tests
    def test_directory_exists(self, download_data):
        """ ensure the directory was created """
        assert os.path.exists(download_data)

    def test_mseeds(self, mseeds):
        """ ensure some files with mseed ext were created """
        assert len(mseeds)

    def test_data_were_downloaded(self, stream_list):
        """ ensure data from the events exists """
        for stream in stream_list:
            assert isinstance(stream, obspy.Stream)
            assert len(stream)


class TestFetchersFromDatasets:
    """ Tests for the fetchers returned from the datasets. """

    @pytest.fixture(scope="class", params=DataSet.datasets)
    def data_fetcher(self, request):
        return obsplus.load_dataset(request.param).get_fetcher()

    def test_type(self, data_fetcher):
        assert isinstance(data_fetcher, Fetcher)

    def test_event_df(self, data_fetcher):
        """ ensure the event df has the event_id column. """
        df = data_fetcher.event_df
        assert "event_id" in df.columns
