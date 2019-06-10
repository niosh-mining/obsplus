"""
A local database for waveform formats.
"""
import os
import time
from collections import defaultdict
from functools import partial, reduce
from itertools import chain
from operator import add
from os.path import abspath
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import obspy
import pandas as pd
import tables
from obspy import UTCDateTime, Stream

import obsplus
from obsplus.bank.core import _Bank
from obsplus.bank.utils import (
    _summarize_trace,
    _IndexCache,
    _summarize_wave_file,
    _try_read_stream,
    get_inventory,
)
from obsplus.constants import (
    NSLC,
    availability_type,
    WAVEFORM_STRUCTURE,
    WAVEFORM_NAME_STRUCTURE,
    utc_time_type,
    get_waveforms_parameters,
)
from obsplus.utils import (
    compose_docstring,
    make_time_chunks,
    to_timestamp,
    get_progressbar,
    thread_lock_function,
    get_nslc_series,
    filter_index,
    replace_null_nlsc_codes,
    _column_contains,
)
from obsplus.waveforms.utils import merge_traces

# from obsplus.interfaces import WaveformClient

# No idea why but this needs to be here to avoid problems with pandas
assert tables.get_hdf5_version()


# ------------------------ constants


class WaveBank(_Bank):
    """
    A class to interact with a directory of waveform files.

    `WaveBank` reads through a directory structure of waveforms files,
    collects info from each one, then creates and index to allow the files
    to be efficiently queried.

    Implements a superset of the :class:`~obsplus.interfaces.WaveformClient`
    interface.

    Parameters
    -------------
    base_path : str
        The path to the directory containing waveform files. If it does not
        exist an empty directory will be created.
    path_structure : str
        Define the directory structure of the wavebank that will be used to
        put waveforms into the directory. Characters are separated by /,
        regardless of operating system. The following words can be used in
        curly braces as data specific variables:
            year, month, day, julday, hour, minute, second, network,
            station, location, channel, time
        example : streams/{year}/{month}/{day}/{network}/{station}
        If no structure is provided it will be read from the index, if no
        index exists the default is {net}/{sta}/{chan}/{year}/{month}/{day}
    name_structure : str
        The same as path structure but for the file name. Supports the same
        variables but requires a period as the separation character. The
        default extension (.mseed) will be added. The default is {time}
        example : {seedid}.{time}
    cache_size : int
        The number of queries to store. Avoids having to read the index of
        the bank multiple times for queries involving the same start and end
        times.
    inventory : obspy.Inventory or str
        obspy.Inventory or path to stationxml to load. Only used to attach
        responses when requested.
    format : str
        The expected format for the waveform files. Any format supported by
        obspy.read is permitted. The default is mseed. Other formats will be
        tried after the default parser fails.
    ext : str or None
        The extension of the waveform files. If provided, only files with
        this extension will be read.
    concurrent_updates
        If True this bank will share an index with other processes, one or
        more of which may perform update_index operations. When used a simple
        file locking mechanism attempts to compensate for shortcomings in
        HDF5 stores lack of concurrency support. This is not needed if all
        processes are only going to read from the bank, nor is it bulletproof,
        but it should help avoid some issues with a few concurrent processes.
    """

    # index columns and types
    metadata_columns = "last_updated path_structure name_structure".split()
    index_str = tuple(NSLC)
    index_float = ("starttime", "endtime")
    index_columns = tuple(list(index_str) + list(index_float) + ["path"])
    columns_no_path = index_columns[:-1]
    gap_columns = tuple(list(columns_no_path) + ["gap_duration"])
    namespace = "/waveforms"
    # other defaults
    buffer = 10.111  # the time before and after the desired times to pull
    # dict defining lengths of str columns (after seed spec)
    # Note: Empty strings get their dtypes caste as S8, which means 8 is the min
    min_itemsize = {"path": 79, "station": 8, "network": 8, "location": 8, "channel": 8}

    # ----------------------------- setup stuff

    def __init__(
        self,
        base_path: Union[str, Path, "WaveBank"] = ".",
        path_structure: Optional[str] = None,
        name_structure: Optional[str] = None,
        cache_size: int = 5,
        inventory: Optional[Union[obspy.Inventory, str]] = None,
        format="mseed",
        ext=None,
        concurrent_updates=False,
    ):
        if isinstance(base_path, WaveBank):
            self.__dict__.update(base_path.__dict__)
            return
        self.format = format
        self.ext = ext
        self.bank_path = abspath(base_path)
        self.inventory = get_inventory(inventory)
        # get waveforms structure based on structures of path and filename
        self.path_structure = path_structure or WAVEFORM_STRUCTURE
        self.name_structure = name_structure or WAVEFORM_NAME_STRUCTURE
        # initialize cache
        self._index_cache = _IndexCache(self, cache_size=cache_size)
        self._concurrent = concurrent_updates

    # ----------------------- index related stuff

    @property
    def last_updated(self) -> Optional[float]:
        """
        Return the last modified time stored in the index, else None.
        """
        self.ensure_bank_path_exists()
        self.block_on_index_lock()
        node = self._time_node
        try:
            out = pd.read_hdf(self.index_path, node)[0]
        except (IOError, IndexError, ValueError, KeyError, AttributeError):
            out = None
        return out

    @property
    def hdf_kwargs(self) -> dict:
        """ A dict of hdf_kwargs to pass to PyTables """
        return dict(
            complib=self._complib,
            complevel=self._complevel,
            format="table",
            data_columns=list(self.index_float),
        )

    @thread_lock_function()
    def update_index(
        self, bar: Optional = None, min_files_for_bar: int = 5000
    ) -> "WaveBank":
        """
        Iterate files in bank and add any modified since last update to index.

        Parameters
        ----------
        bar
            An class that has an `update` and `finish` method, should behave
            the same as the progressbar.ProgressBar class. This method provides
            a way to override the default progress bar and is used only for
            hooking this class into larger (graphical) systems.
        min_files_for_bar
            Minimum number of un-indexed files required for using the
            progress bar.
        """
        self._enforce_min_version()
        num_files = sum([1 for _ in self._unindexed_file_iterator()])
        if num_files >= min_files_for_bar:
            print(f"updating or creating waveform index for {self.bank_path}")
        kwargs = {"min_value": min_files_for_bar, "max_value": num_files}
        # init progress bar
        bar = get_progressbar(**kwargs) if bar is None else bar(**kwargs)
        update_time = time.time()
        # loop over un-index files and add info to index
        updates = []
        for num, fi in enumerate(self._unindexed_file_iterator()):
            updates.append(_summarize_wave_file(fi, format=self.format))
            # update bar
            if bar and num % self._bar_update_interval == 0:
                bar.update(num)
        getattr(bar, "finish", lambda: None)()  # call finish if applicable

        if len(updates):  # flatten list and make df
            with self.lock_index():
                self._write_update(list(chain.from_iterable(updates)), update_time)
            # clear cache out when new traces are added
            self._index_cache.clear_cache()
        return self

    def _write_update(self, updates, update_time):
        """ convert updates to dataframe, then append to index table """
        # read in dataframe and cast to correct types
        df = pd.DataFrame.from_dict(updates)
        # ensure the bank path is not in the path column
        df["path"] = df["path"].str.replace(self.bank_path, "")
        # assert not df.duplicated().any(), "update index has duplicate entries"
        for str_index in self.index_str:
            sser = df[str_index].astype(str)
            df[str_index] = sser.str.replace("b", "").str.replace("'", "")
        for float_index in self.index_float:
            df[float_index] = df[float_index].astype(float)
        # populate index store and update metadata
        assert not df.isnull().any().any(), "null values found in index dataframe"
        with pd.HDFStore(self.index_path) as store:
            node = self._index_node
            try:
                nrows = store.get_storer(node).nrows
            except (AttributeError, KeyError):
                store.append(
                    node, df, min_itemsize=self.min_itemsize, **self.hdf_kwargs
                )
            else:
                df.index += nrows
                store.append(node, df, append=True, **self.hdf_kwargs)
            # update timestamp
            update_time = time.time() if update_time is None else update_time
            store.put(self._time_node, pd.Series(update_time))
            # make sure meta table also exists.
            # Note this is hear to avoid opening the store again.
            if self._meta_node not in store:
                meta = self._make_meta_table()
                store.put(self._meta_node, meta, format="table")

    def _ensure_meta_table_exists(self):
        """
        If the bank path exists ensure it has a meta table, if not create it.
        """
        if not Path(self.index_path).exists():
            return
        with self.lock_index():
            with pd.HDFStore(self.index_path) as store:
                # add metadata if not in store
                if self._meta_node not in store:
                    meta = self._make_meta_table()
                    store.put(self._meta_node, meta, format="table")

    @compose_docstring(waveform_params=get_waveforms_parameters)
    def read_index(
        self,
        network: Optional[str] = None,
        station: Optional[str] = None,
        location: Optional[str] = None,
        channel: Optional[str] = None,
        starttime: Optional[utc_time_type] = None,
        endtime: Optional[utc_time_type] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Return a dataframe of the index, optionally applying filters.

        Parameters
        ----------
        {waveform_params}
        kwargs
            kwargs are passed to pandas.read_hdf function
        """
        self.ensure_bank_path_exists()
        if starttime is not None and endtime is not None:
            if starttime > endtime:
                msg = f"starttime cannot be greater than endtime"
                raise ValueError(msg)
        if not os.path.exists(self.index_path):
            self.update_index()
        # if no file was created (dealing with empty bank) return empty index
        if not os.path.exists(self.index_path):
            return pd.DataFrame(columns=self.index_columns)
        # grab index from cache
        index = self._index_cache(starttime, endtime, buffer=self.buffer, **kwargs)
        # filter and return
        filt = filter_index(
            index, network=network, station=station, location=location, channel=channel
        )
        return index[filt]

    def _read_metadata(self):
        """
        Read the metadata table.
        """
        self.block_on_index_lock()
        try:
            return pd.read_hdf(self.index_path, self._meta_node)
        except (FileNotFoundError, ValueError, KeyError):
            self._ensure_meta_table_exists()
            return pd.read_hdf(self.index_path, self._meta_node)

    # ------------------------ availability stuff

    @compose_docstring(get_waveform_params=get_waveforms_parameters)
    def get_availability_df(self, *args, **kwargs) -> pd.DataFrame:
        """
        Return a dataframe specifying the availability of the archive.

        Parameters
        ----------
        {get_waveform_params}

        """
        # no need to read in path, just read needed columns
        ind = self.read_index(*args, columns=self.columns_no_path, **kwargs)
        gro = ind.groupby(list(NSLC))
        min_start = gro.starttime.min().reset_index()
        max_end = gro.endtime.max().reset_index()
        return pd.merge(min_start, max_end)

    def availability(
        self,
        network: str = None,
        station: str = None,
        location: str = None,
        channel: str = None,
    ) -> availability_type:
        df = self.get_availability_df(network, station, location, channel)
        # convert timestamps to UTCDateTime objects
        df["starttime"] = df.starttime.apply(UTCDateTime)
        df["endtime"] = df.endtime.apply(UTCDateTime)
        # convert to list of tuples, return
        return df.to_records(index=False).tolist()

    # --------------------------- get gaps stuff

    def _get_gap_dfs(self, df, min_gap):
        """ function to apply to each group of seed_id dataframes """
        dd = df.sort_values(["starttime", "endtime"]).reset_index(drop=True)
        shifted_starttimes = dd.starttime.shift(-1)
        gap_index: pd.DataFrame = (dd.endtime + min_gap) < shifted_starttimes
        # create a dataframe of gaps
        df = dd[gap_index]
        df["starttime"] = dd.endtime[gap_index]
        df["endtime"] = shifted_starttimes[gap_index]
        df["gap_duration"] = df["endtime"] - df["starttime"]
        return df

    @compose_docstring(get_waveforms_params=get_waveforms_parameters)
    def get_gaps_df(self, *args, min_gap=1.0, **kwargs) -> pd.DataFrame:
        """
        Return a dataframe containing an entry for every gap in the archive.

        Parameters
        ----------
        {get_waveforms_params}
        min_gap
            The minimum gap to report. Should at least be greater than
            the sample rate in order to avoid counting one sample between
            files as gaps. If files have some overlaps this parameter can
            be set to 0.

        Returns
        -------
        pd.DataFrame
        """
        index = self.read_index(*args, columns=self.columns_no_path, **kwargs)
        group = index.groupby(list(NSLC))
        func = partial(self._get_gap_dfs, min_gap=min_gap)
        out = group.apply(func).reset_index(drop=True)
        if out.empty:  # if not gaps return empty dataframe with needed cols
            return pd.DataFrame(columns=self.gap_columns)
        return out

    @compose_docstring(get_waveforms_params=get_waveforms_parameters)
    def get_uptime_df(self, *args, **kwargs) -> pd.DataFrame:
        """
        Return a dataframe with uptime stats for selected channels.

        Parameters
        ----------
        {get_waveforms_params}

        """
        # get total number of seconds bank spans for each seed id
        avail = self.get_availability_df(*args, **kwargs)
        avail["duration"] = avail["endtime"] - avail["starttime"]
        # get total duration of gaps by seed id
        gaps_df = self.get_gaps_df(*args, **kwargs)
        if not gaps_df.empty:
            gap_totals = gaps_df.groupby(list(NSLC)).gap_duration.sum()
            gap_total_df = pd.DataFrame(gap_totals).reset_index()
        else:
            gap_total_df = pd.DataFrame(avail[list(NSLC)])
            gap_total_df["gap_duration"] = 0.0
        # merge gap dataframe with availability dataframe, add uptime and %
        df = pd.merge(avail, gap_total_df)
        df["uptime"] = df.duration - df.gap_duration
        df["availability"] = df["uptime"] / df["duration"]
        return df

    # ------------------------ get waveform related methods

    def get_waveforms_bulk(
        self, bulk: List[str], index: Optional[pd.DataFrame] = None, **kwargs
    ) -> Stream:
        """
        Get a large number of waveforms with a bulk request.

        Parameters
        ----------
        bulk
            A list of any number of lists containing the following:
            (network, station, location, channel, starttime, endtime).
        index
            A dataframe returned by read_index. Enables calling code to only
            read the index from disk once for repetitive calls.
        """
        if not bulk:  # return emtpy waveforms if empty list or None
            return obspy.Stream()

        def _func(time, ind, df):
            """ return waveforms from df of bulk parameters """
            match_chars = {"*", "?", "[", "]"}
            t1, t2 = time[0], time[1]
            # filter index based on start/end times
            in_time = ~((ind["starttime"] > t2) | (ind["endtime"] < t1))
            ind = ind[in_time]
            # create indices used to load data
            ar = np.ones(len(ind))  # indices of ind to use to load data
            df = df[(df.t1 == time[0]) & (df.t2 == time[1])]
            # determine which columns use any matching or other select features
            uses_matches = [_column_contains(df[x], match_chars) for x in NSLC]
            match_ar = np.array(uses_matches).any(axis=0)
            df_match = df[match_ar]
            df_no_match = df[~match_ar]
            # handle columns that need matches (more expensive)
            if not df_match.empty:
                match_bulk = df_match.to_records(index=False)
                mar = np.array([filter_index(ind, *tuple(b)[:4]) for b in match_bulk])
                ar = np.logical_and(ar, mar.any(axis=0))
            # handle columns that do not need matches
            if not df_no_match.empty:
                nslc1 = set(get_nslc_series(df_no_match))
                nslc2 = get_nslc_series(ind)
                ar = np.logical_and(ar, nslc2.isin(nslc1))
            return self._index2stream(ind[ar], t1, t2)

        # get a dataframe of the bulk arguments, convert time to float
        df = pd.DataFrame(bulk, columns=list(NSLC) + ["utc1", "utc2"])
        df["t1"] = df["utc1"].apply(float)
        df["t2"] = df["utc2"].apply(float)
        # read index that contains any times that might be used, or filter
        # provided index
        t1, t2 = df["t1"].min(), df["t2"].max()
        if index is not None:
            ind = index[~((index.starttime > t2) | (index.endtime < t1))]
        else:
            ind = self.read_index(starttime=t1, endtime=t2)
        # groupby.apply calls two times for each time set, avoid this.
        unique_times = np.unique(df[["t1", "t2"]].values, axis=0)
        streams = [_func(time, df=df, ind=ind) for time in unique_times]
        return reduce(add, streams)

    @compose_docstring(get_waveforms_params=get_waveforms_parameters)
    def get_waveforms(
        self,
        network: Optional[str] = None,
        station: Optional[str] = None,
        location: Optional[str] = None,
        channel: Optional[str] = None,
        starttime: Optional[obspy.UTCDateTime] = None,
        endtime: Optional[obspy.UTCDateTime] = None,
        attach_response: bool = False,
    ) -> Stream:
        """
        Get waveforms from the bank.

        Parameters
        ----------
        {get_waveforms_params}
        attach_response : bool
            If True attach the response to the waveforms using the stations

        Notes
        -----
        All string parameters can use posix style matching with * and ? chars.
        All datapoints between selected starttime and endtime will be returned.
        Consequently there may be gaps in the returned stream.
        """
        index = self.read_index(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
        )
        return self._index2stream(index, starttime, endtime, attach_response)

    @compose_docstring(get_waveforms_params=get_waveforms_parameters)
    def yield_waveforms(
        self,
        network: Optional[str] = None,
        station: Optional[str] = None,
        location: Optional[str] = None,
        channel: Optional[str] = None,
        starttime: Optional[obspy.UTCDateTime] = None,
        endtime: Optional[obspy.UTCDateTime] = None,
        attach_response: bool = False,
        duration: float = 3600.0,
        overlap: Optional[float] = None,
    ) -> Stream:
        """
        Yield waveforms from the bank.

        Parameters
        ----------
        {get_waveforms_params}
        attach_response : bool
            If True attach the response to the waveforms using the stations
        duration : float
            The duration of the streams to yield. All channels selected
            channels will be included in the waveforms.
        overlap : float
            If duration is used, the amount of overlap in yielded streams,
            added to the end of the waveforms.


        Notes
        -----
        All string parameters can use posix style matching with * and ? chars.
        """
        # get times in float format
        starttime = to_timestamp(starttime, 0.0)
        endtime = to_timestamp(endtime, "2999-01-01")
        # read in the whole index df
        index = self.read_index(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
        )
        # adjust start/end times
        starttime = max(starttime, index.starttime.min())
        endtime = min(endtime, index.endtime.max())
        # chunk time and iterate over chunks
        time_chunks = make_time_chunks(starttime, endtime, duration, overlap)
        for t1, t2 in time_chunks:
            con1 = (index.starttime - self.buffer) > t2
            con2 = (index.endtime + self.buffer) < t1
            ind = index[~(con1 | con2)]
            if not len(ind):
                continue
            yield self._index2stream(ind, t1, t2, attach_response)

    def get_waveforms_by_seed(
        self,
        seed_id: Union[List[str], str],
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        attach_response: bool = False,
    ) -> Stream:
        """
        Get waveforms based on a single seed_id or a list of seed_ids.

        Seed ids have the following form: network.station.location.channel,
        it does not yet support usage of wildcards.

        Parameters
        ----------
        seed_id
            A single seed id or sequence of ids
        starttime
            The beginning of time to pull
        endtime
            The end of the time to pull
        attach_response
            If True, and if a an stations is attached to the bank, attach
            the response to the waveforms before returning.
        """
        seed_id = [seed_id] if isinstance(seed_id, str) else seed_id
        index = self._read_index_by_seed(seed_id, starttime, endtime)
        return self._index2stream(index, starttime, endtime, attach_response)

    def _read_index_by_seed(self, seed_id, starttime, endtime):
        """ read the index by seed_ids """
        if not os.path.exists(self.index_path):
            self.update_index()
        index = self._index_cache(starttime, endtime, buffer=self.buffer)
        seed = (
            index.network
            + "."
            + index.station
            + "."
            + index.location
            + "."
            + index.channel
        )
        return index[seed.isin(seed_id)]

    # ----------------------- deposit waveforms methods

    def put_waveforms(self, stream: obspy.Stream, name=None, update_index=True):
        """
        Add the waveforms in a waveforms to the bank.

        Parameters
        ----------
        stream
            An obspy waveforms object to add to the bank
        name
            Name of file, if None it will be determined based on contents
        update_index
            Flag to indicate whether or not to update the waveform index
            after writing the new events. Default is True.
        """
        self.ensure_bank_path_exists(create=True)
        st_dic = defaultdict(lambda: [])
        # iter the waveforms and group by common paths
        for tr in stream:
            summary = _summarize_trace(
                tr,
                name=name,
                path_struct=self.path_structure,
                name_struct=self.name_structure,
            )
            path = os.path.join(self.bank_path, summary["path"])
            st_dic[path].append(tr)
        # iter all the unique paths and save
        for path, tr_list in st_dic.items():
            # make the dir structure of it doesn't exist
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            stream = obspy.Stream(traces=tr_list)
            # load the waveforms if the file already exists
            if os.path.exists(path):
                st_existing = obspy.read(path)
                stream += st_existing
            # polish streams and write
            stream.merge(method=1)
            stream.write(path, format="mseed")
        # update the index as the contents have changed
        if st_dic and update_index:
            self.update_index()

    # ------------------------ misc methods

    def _index2stream(
        self, index, starttime=None, endtime=None, attach_response=False
    ) -> Stream:
        """ return the waveforms in the index """
        # get abs path to each datafame
        files: pd.Series = (self.bank_path + index.path).unique()
        # iterate the files to read and try to load into waveforms
        stt = obspy.Stream()
        kwargs = dict(
            format=self.format,
            starttime=obspy.UTCDateTime(starttime) if starttime else None,
            endtime=obspy.UTCDateTime(endtime) if endtime else None,
        )
        for st in (_try_read_stream(x, **kwargs) for x in files):
            if st is not None and len(st):
                stt += st
        # sort out nullish nslc codes
        stt = replace_null_nlsc_codes(stt)
        # filter out any traces not in index (this can happen when files hold
        # multiple traces).
        nslc = set(get_nslc_series(index))
        stt.traces = [x for x in stt if x.id in nslc]
        # trim, merge, attach response
        stt = self._prep_output_stream(stt, starttime, endtime, attach_response)
        return stt

    def _prep_output_stream(
        self, st, starttime=None, endtime=None, attach_response=False
    ) -> obspy.Stream:
        """
        Prepare waveforms object for output by trimming to desired times,
        merging channels, and attaching responses.
        """
        if not len(st):
            return st
        starttime = starttime or min([x.stats.starttime for x in st])
        endtime = endtime or max([x.stats.endtime for x in st])
        # trim
        st.trim(starttime=UTCDateTime(starttime), endtime=UTCDateTime(endtime))
        if attach_response:
            st.attach_response(self.inventory)
        return merge_traces(st, inplace=True).sort()

    def get_service_version(self):
        """ Return the version of obsplus """
        return obsplus.__version__


# --- auxiliary functions
