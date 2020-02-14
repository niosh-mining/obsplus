"""
The test TA dataset.
"""
import obspy

from obsplus import WaveBank
from obsplus.datasets.dataset import DataSet
from obsplus.utils.time import make_time_chunks


class TA(DataSet):
    """
    A small dataset from the TA with low sampling rate channels.

    Contains about 10 days of data from two stations: M11A and M14A.
    """

    name = "ta_test"
    version = "0.0.0"

    level = "response"
    starttime = obspy.UTCDateTime("2007-02-15")
    endtime = obspy.UTCDateTime("2007-02-25")

    bulk = [
        ("TA", "M11A", "*", "VH*", starttime, endtime),
        ("TA", "M14A", "*", "VH*", starttime, endtime),
    ]

    def download_stations(self):
        """Download TA stations."""
        inv_path = self.station_path / "stations.xml"
        inv_path.parent.mkdir(exist_ok=True, parents=True)
        inv = self._download_client.get_stations_bulk(self.bulk, level=self.level)
        inv.write(str(inv_path), "stationxml")

    def download_waveforms(self):
        """Download TA waveforms."""
        st = self._download_client.get_waveforms_bulk(self.bulk)
        self.build_archive(st)
        # update the index
        WaveBank(self.waveform_path).update_index()

    def build_archive(self, st, starttime=None, endtime=None):
        """
        Build the archive.
        """
        starttime = starttime or self.starttime
        endtime = endtime or self.endtime

        for utc1, utc2 in make_time_chunks(starttime, endtime, duration=3600):
            stt = st.copy()
            stt.trim(starttime=utc1, endtime=utc2)
            channels = {x.stats.channel for x in st}
            for channel in channels:
                ts = stt.select(channel=channel)
                if not len(ts):
                    continue
                stats = ts[0].stats
                net, sta = stats.network, stats.station
                loc, cha = stats.location, stats.channel
                utc_str = str(utc1).split(".")[0].replace(":", "-")
                end = utc_str + ".mseed"
                fpath = self.waveform_path / net / sta / loc / cha / end
                fpath.parent.mkdir(parents=True, exist_ok=True)
                ts.write(str(fpath), "mseed")
