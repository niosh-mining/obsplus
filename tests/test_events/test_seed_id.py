#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:35:58 2019

@author: shawn
"""

import pytest

import obspy
import obsplus


# I'm not sure of the best place to put these tests, feel free to move them
class TestGetSeedId:
    """Tests for the get_seed_id function"""

    def test_get_seed_id(self):
        """Make sure it is possible to retrieve the seed id"""
        wid = obspy.core.event.WaveformStreamID(
            network_code="AA",
            station_code="BBB",
            location_code="CC",
            channel_code="DDD",
        )
        pick = obspy.core.event.Pick(waveform_id=wid)
        amp = obspy.core.event.Amplitude(pick_id=pick.resource_id)
        station_mag = obspy.core.event.StationMagnitude(amplitude_id=amp.resource_id)
        station_mag_cont = obspy.core.event.StationMagnitudeContribution(
            station_magnitude_id=station_mag.resource_id
        )
        seed = obsplus.events.utils.get_seed_id(station_mag_cont)
        assert seed == "AA.BBB.CC.DDD"

    def test_no_seed_id(self):
        """Make sure raises AttributeError if no seed info found"""
        with pytest.raises(AttributeError):
            obsplus.events.utils.get_seed_id(obspy.core.event.Pick())

    def test_unsupported(self):
        """Make sure an unsupported object raises TypeError"""
        with pytest.raises(TypeError):
            obsplus.events.utils.get_seed_id(obspy.core.event.Origin())
