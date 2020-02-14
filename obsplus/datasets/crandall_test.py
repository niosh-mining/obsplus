"""
The Crandall Canyon dataset
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


class Crandall(DataSet):
    """
    Seismic data recorded by regional stations associated with the Crandall
    Canyon Mine Collapse (goo.gl/2ezyZD). Six mine workers and three rescue
    workers lost their lives in the disaster.

    This dataset is of interest to mine seismology researchers because better
    understanding of the conditions and sources involved will help prevent
    future disasters.

    I have personally modified and added some phase picks to the events.
    As such, this dataset should be used for testing and demonstration
    purposes only.
    """

    name = "crandall_test"
    version = "0.0.1"
    # Days of interest. The collapse occurred on Aug 6th
    starttime = obspy.UTCDateTime("2007-08-06")
    endtime = obspy.UTCDateTime("2007-08-10")
    time_before = np.timedelta64(10, "s")
    time_after = np.timedelta64(60, "s")
    # approx. location of collapse
    latitude = 39.462
    longitude = -111.228
    max_dist = 150  # max distance from event (km) to keep station

    def download_events(self):
        """ Just copy the events into a directory. """
        cat = obspy.read_events(str(self.source_path / "events.xml"))
        obsplus.EventBank(self.event_path).put_events(cat)

    def _download_crandall(self):
        """ download waveform/station info for dataset. """
        bank = WaveBank(self.waveform_path)
        domain = CircularDomain(
            self.latitude,
            self.longitude,
            minradius=0,
            maxradius=kilometers2degrees(self.max_dist),
        )
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
                channel_priorities=["HH[ZNE]", "BH[ZNE]"],
                location_priorities=["", "00", "01", "--"],
            )
            kwargs = dict(
                domain=domain,
                restrictions=restrictions,
                mseed_storage=str(self.waveform_path),
                stationxml_storage=str(self.station_path),
            )
            MassDownloader(providers=[self._download_client]).download(**kwargs)
            # ensure data have actually been downloaded
            bank.update_index()
            assert not bank.read_index(starttime=starttime, endtime=endtime).empty

    download_stations = _download_crandall
    download_waveforms = _download_crandall
