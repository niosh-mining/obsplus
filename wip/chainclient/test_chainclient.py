"""
Tests for chain client
"""

import obspy
import pytest

import obsplus
import pandas as pd
from obsplus.structures.chainclient import ChainClient
import copy


class TestChainClientBasic:
    """ test the chain client using standard python objects """

    @pytest.fixture
    def client_chain(self, bing_fetcher, ta_archive):
        wclients = [bing_fetcher.waveform_client, obsplus.WaveBank(ta_archive)]
        kwargs = dict(
            event_clients=[bing_fetcher.event_client.get_events()],
            station_clients=bing_fetcher.station_client.get_stations(),
            waveform_clients=wclients,
        )

        return ChainClient(**kwargs)

    @pytest.fixture
    def kem_stream(self, client_chain):
        """ return a waveforms from M17A """
        t1 = obspy.UTCDateTime("2009-04-02T00-00-00")
        t2 = obspy.UTCDateTime("2009-04-02T02-00-00")
        return client_chain.get_waveforms("TA", "M17A", "*", "BHZ", t1, t2)

    @pytest.fixture
    def backup_stream(self, client_chain):
        """ ask for a waveforms only found in backup client (clients) """
        t1 = obspy.UTCDateTime("2007-02-15")
        t2 = obspy.UTCDateTime("2007-02-16")
        return client_chain.get_waveforms("TA", "M11A", "*", "VHZ", t1, t2)

    @pytest.fixture
    def waveform_bulk(self, client_chain):
        """ make a get_waveforms bulk request """
        t1a = obspy.UTCDateTime("2007-02-15")
        t2a = obspy.UTCDateTime("2007-02-16")
        t3a = obspy.UTCDateTime("2007-02-17")
        t1b = obspy.UTCDateTime("2009-04-02T00-00-00")
        t2b = obspy.UTCDateTime("2009-04-02T02-00-00")
        bulk = [
            ("TA", "M11A", "*", "*", t1a, t2a),
            ("TA", "M14A", "*", "*", t1a, t2a),
            ("TA", "M11A", "*", "*", t2a, t3a),
            ("TA", "M14A", "*", "*", t2a, t3a),
            ("TA", "M17A", "*", "*", t1b, t2b),
            ("UR", "R2D2", "*", "WEE", t1a, t2a),  # doesnt exist
        ]
        # import pdb; pdb.set_trace()
        return client_chain.get_waveforms_bulk(bulk)

    # tests
    def test_stream(self, bing_stream):
        """ ensure the M17A waveforms is a waveforms and has a length """
        assert len(bing_stream)
        assert isinstance(bing_stream, obspy.Stream)

    def test_backup_stream(self, backup_stream):
        """ ask for data only found in clients attr to make when it is not
        found in other clients the backup is used """
        assert len(backup_stream)
        assert isinstance(backup_stream, obspy.Stream)

    def test_get_waveforms_bulk(self, waveform_bulk):
        """ ensure bulk waveforms worked """
        expected_stations = {"M11A", "M14A", "M17A"}
        got_stations = {tr.stats.station for tr in waveform_bulk}
        assert expected_stations == got_stations

    def test_get_attr(self, client_chain):
        """ ensure get attr falls back to clients """
        df = client_chain.read_index()
        assert isinstance(df, pd.DataFrame)
        assert len(df)

    def test_can_deep_copy(self, client_chain):
        """ ensure deep copy doesnt raise """
        try:
            copy.deepcopy(client_chain)
        except Exception as e:
            pytest.fail("deep copy raised exception")
