"""
The default test dataset. This consists exclusively of obspy's default objects
saved to disk.
"""
from pathlib import Path

import obspy

import obsplus
from obsplus.datasets.dataset import DataSet


class Default(DataSet):
    """
    A dataset from obspy's default stream, catalog, and inventory.

    Each object is simply saved to the appropriate directory.
    """

    name = "default_test"
    version = "0.0.0"

    # Days of interest. The collapse occurred on Aug 6th

    def download_events(self):
        """ Just copy the events into a directory. """
        cat = obspy.read_events()
        obsplus.EventBank(self.event_path).put_events(cat)

    def download_stations(self):
        """Copy the default inventory to a directory."""
        inv = obspy.read_inventory()
        path = Path(self.station_path) / "inventory.xml"
        path.parent.mkdir(exist_ok=True, parents=True)
        inv.write(str(path), "stationxml")

    def download_waveforms(self):
        """Copy the default Stream into a directory."""
        st = obspy.read()
        obsplus.WaveBank(self.waveform_path).put_waveforms(st)
