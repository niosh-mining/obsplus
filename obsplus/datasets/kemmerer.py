"""
The Kemmerer dataset.
"""

import obspy
from obspy.clients.fdsn.mass_downloader import (
    RectangularDomain,
    Restrictions,
    MassDownloader,
)

from obsplus import EventBank
from obsplus.datasets.dataloader import DataSet, base_path
from obsplus.events.utils import catalog_to_directory


class Kemmerer(DataSet):
    """
    Another small TA dataset based on stations near Kemmerer Wyoming.

    Contains stations M17A and M18A as well as some recorded surface blasts
    from a nearby surface coal mine.
    """

    name = "kemmerer"
    version = "0.0.0"

    bulk = [("TA", "M17A", "*", "BH?"), ("AT", "M18A", "*", "BH?")]

    def download_events(self):
        """ Simply copy events from base directory. """
        cat_path = base_path / self.name / "events.xml"
        assert cat_path.exists(), "this should ship with obsplus"
        cat = obspy.read_events(str(cat_path))
        catalog_to_directory(cat, self.event_path)
        # update index
        EventBank(self.event_path).update_index()

    def _download_kemmerer(self):
        """ downloads both stations and """
        for station in ["M17A", "M18A"]:
            domain = RectangularDomain(
                minlatitude=40.0,
                maxlatitude=43.0,
                minlongitude=-111.0,
                maxlongitude=-110.0,
            )
            restrictions = Restrictions(
                starttime=obspy.UTCDateTime("2009-04-01T00:00:00"),
                endtime=obspy.UTCDateTime("2009-04-04T00:00:00"),
                chunklength_in_sec=3600,
                network="TA",
                channel="BH?",
                station=station,
                reject_channels_with_gaps=False,
                minimum_length=0.0,
                minimum_interstation_distance_in_m=10.0,
            )
            MassDownloader(providers=[self._download_client]).download(
                domain,
                restrictions,
                mseed_storage=str(self.waveform_path),
                stationxml_storage=str(self.station_path),
            )

    download_stations = _download_kemmerer
    download_waveforms = _download_kemmerer
