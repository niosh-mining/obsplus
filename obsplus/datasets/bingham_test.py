"""
The Bingham dataset.
"""
import numpy as np
import obspy
from obspy import UTCDateTime as UTC
from obspy.clients.fdsn.mass_downloader import (
    CircularDomain,
    Restrictions,
    MassDownloader,
)
from obspy.geodetics import kilometers2degrees

import obsplus
from obsplus import WaveBank, events_to_df
from obsplus.datasets.dataset import DataSet


class Bingham(DataSet):
    """
    The Bingham Canyon dataset includes waveforms recorded during and after
    the Manefay Slide, one of the largest anthropogenic landslides ever
    recorded (https://bit.ly/2bsRsyR). Fortunately, due to close monitoring,
    no one was injured.

    I have personally modified and added some phase picks to the events.
    As such, this dataset should be used for testing and demonstration
    purposes only.
    """

    name = "bingham_test"
    version = "0.0.0"
    time_before = np.timedelta64(10, "s")
    time_after = np.timedelta64(60, "s")
    # define spatial extents variables (center of pit)
    latitude = 40.53829
    longitude = -112.149_506
    max_dist = 20  # distance in km

    def download_events(self):
        """ Simply copy events from base directory. """
        cat = obspy.read_events(str(self.source_path / "events.xml"))
        obsplus.EventBank(self.event_path).put_events(cat)

    def _download_bingham(self):
        """ Use obspy's mass downloader to get station/waveforms data. """
        bank = WaveBank(self.waveform_path)
        domain = CircularDomain(
            self.latitude,
            self.longitude,
            minradius=0,
            maxradius=kilometers2degrees(self.max_dist),
        )
        chan_priorities = ["HH[ZNE]", "BH[ZNE]", "EL[ZNE]", "EN[ZNE]"]
        cat = obspy.read_events(str(self.source_path / "events.xml"))
        df = events_to_df(cat)
        for _, row in df.iterrows():
            starttime = row.time - self.time_before
            endtime = row.time + self.time_after
            restrictions = Restrictions(
                starttime=UTC(starttime),
                endtime=UTC(endtime),
                minimum_length=0.90,
                minimum_interstation_distance_in_m=100,
                channel_priorities=chan_priorities,
                location_priorities=["", "00", "01", "--"],
            )
            kwargs = dict(
                domain=domain,
                restrictions=restrictions,
                mseed_storage=str(self.waveform_path),
                stationxml_storage=str(self.station_path),
            )
            MassDownloader(providers=[self._download_client]).download(**kwargs)
            # ensure data were downloaded
            bank.update_index()
            assert not bank.read_index(starttime=starttime, endtime=endtime).empty

        # update wavebank
        WaveBank(self.waveform_path).update_index()

    download_stations = _download_bingham
    download_waveforms = _download_bingham
