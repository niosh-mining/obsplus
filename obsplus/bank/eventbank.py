"""
Class for interacting with events on a filesystem.
"""

import time
import warnings
from functools import reduce
from operator import add
from os.path import exists
from os.path import getmtime, abspath
from pathlib import Path
from typing import Optional, Union

import obspy
import obspy.core.event as ev
import pandas as pd

import obsplus
import obsplus.events.pd
from obsplus.bank.core import _Bank
from obsplus.bank.utils import (
    _IndexCache,
    _summarize_event,
    sql_connection,
    _read_table,
    _get_tables,
    _drop_rows,
)
from obsplus.constants import (
    EVENT_PATH_STRUCTURE,
    EVENT_NAME_STRUCTURE,
    EVENT_DTYPES,
    get_events_parameters,
)
from obsplus.exceptions import BankDoesNotExistError
from obsplus.utils import (
    try_read_catalog,
    get_progressbar,
    thread_lock_function,
    compose_docstring,
)

# --- define static types

# Fixed types for expected columns
COLUMN_TYPES = dict(EVENT_DTYPES)
COLUMN_TYPES.pop("stations", None)
COLUMN_TYPES["path"] = str
STR_COLUMNS = {i for i, v in COLUMN_TYPES.items() if issubclass(v, str)}

# unsupported query options

UNSUPPORTED_QUERY_OPTIONS = {"minradius", "maxradius"}


# int_cols = {key for key, val in column_types.items() if val is int}


class EventBank(_Bank):
    """
    A class to interact with a directory of event files.

    Event bank reads through a directory structure of event files,
    collects info from each one, then creates and index to allow the files
    to be efficiently queried.

    Implements a superset of the :class:`~obsplus.interfaces.EventClient`
    interface.

    Parameters
    ----------
    base_path
        The path to the directory containing event files. If it does not
        exist an empty directory will be created.
    path_structure
        Define the directory structure used by the event bank. Characters are
        separated by /, regardless of operating system. The following
        words can be used in curly braces as data specific variables:
            year, month, day, julday, hour, minute, second, event_id,
            event_id_short
        If no structure is provided it will be read from the index, if no
        index exists the default is {year}/{month}/{day}
    name_structure : str
        The same as path structure but for the file name. Supports the same
        variables and a slash cannot be used in a file name on most operating
        systems. The default extension (.xml) will be added.
        The default is {time}_{event_id_short}.
    format
        The anticipated format of the event files. Any format supported by the
        obspy.read_events function is permitted.
    ext
        The extension on the files. Can be used to avoid parsing non-event
        files.
    cache_size : int
        The number of queries to store. Avoids having to read the index of
        the database multiple times for queries involving the same start and
        end times.
    """

    namespace = "/events"
    index_name = ".index.db"  # name of index file

    def __init__(
        self,
        base_path: Union[str, Path, "EventBank"] = ".",
        path_structure: Optional[str] = None,
        name_structure: Optional[str] = None,
        cache_size: int = 5,
        format="quakeml",
        ext=".xml",
    ):
        """ Initialize an instance. """
        if isinstance(base_path, EventBank):
            self.__dict__.update(base_path.__dict__)
            return
        self.bank_path = abspath(base_path)
        self._index = None
        self.format = format
        self.ext = ext
        # get waveforms structure based on structures of path and filename
        ps = path_structure or self._path_structure or EVENT_PATH_STRUCTURE
        self.path_structure = ps
        ns = name_structure or self._name_structure or EVENT_NAME_STRUCTURE
        self.name_structure = ns
        # initialize cache
        self._index_cache = _IndexCache(self, cache_size=cache_size)

    @property
    def last_updated(self):
        """ Return the last modified time stored in the index, else 0.0 """
        with sql_connection(self.index_path) as con:
            try:
                return _read_table(self._time_node, con).loc[0, "time"]
            except (pd.io.sql.DatabaseError, KeyError):  # table is empty
                return 0.0

    @property
    def _path_structure(self):
        """ return the path structure stored in memory """
        try:
            return self._read_metadata()["path_structure"][0]
        except (pd.io.sql.DatabaseError, BankDoesNotExistError):
            return None

    @property
    def _name_structure(self):
        """ return the name structure stored in memory """
        try:
            return self._read_metadata()["name_structure"][0]
        except (pd.io.sql.DatabaseError, BankDoesNotExistError):
            return None

    # --- index stuff

    @compose_docstring(get_events_params=get_events_parameters)
    def read_index(self, **kwargs) -> pd.DataFrame:
        """
        Read the index and return a dataframe containing the event info.

        Parameters
        ----------
        {get_events_params}
        """
        self.ensure_bank_path_exists()
        if set(kwargs) & UNSUPPORTED_QUERY_OPTIONS:
            unsupported_options = set(kwargs) & UNSUPPORTED_QUERY_OPTIONS
            msg = f"Query parameters {unsupported_options} are not supported"
            raise ValueError(msg)
        with sql_connection(self.index_path) as con:
            try:
                df = _read_table(self._index_node, con, **kwargs).set_index("event_id")
            except pd.io.sql.DatabaseError:  # empty or no db, return empty index
                df = pd.DataFrame(columns=list(COLUMN_TYPES)).set_index("event_id")
        # coerce datatypes
        dtype = {i: COLUMN_TYPES[i] for i in set(COLUMN_TYPES) & set(df.columns)}
        df = df.astype(dtype=dtype)
        # replace "None" with None on str columns
        str_cols = STR_COLUMNS & set(df.columns)
        df.loc[:, str_cols] = df.loc[:, str_cols].replace(["None"], [None])
        return df

    @thread_lock_function()
    def update_index(
        self, bar: Optional = None, min_files_for_bar: int = 100
    ) -> "EventBank":
        """
        Iterate files in bank and add any modified since last update to index.

        Parameters
        ----------
        bar
            An class that has an `update` and `finish` method, should behave
            the same as the progressbar.ProgressBar class. This method provides
            a way to override the default progress bar but is rarely needed.
        min_files_for_bar
            Minimum number of un-indexed files required for displaying the
            progress bar.
        """
        self._enforce_min_version()
        num_files = sum([1 for _ in self._unindexed_file_iterator()])
        if num_files >= min_files_for_bar:
            print(f"updating or creating event index for {self.bank_path}")
        kwargs = {"min_value": min_files_for_bar, "max_value": num_files}
        bar = get_progressbar(**kwargs) if bar is None else bar(**kwargs)
        # loop over un-index files and update
        events, update_time, paths = [], [], []
        for num, fi in enumerate(self._unindexed_file_iterator()):
            cat = try_read_catalog(fi, format=self.format)
            if cat is None:
                continue
            for event in cat:
                events.append(event)
                update_time.append(getmtime(fi))
                paths.append(fi.replace(self.bank_path, ""))
            # update progress bar
            if bar and num % self._bar_update_interval == 0:
                bar.update(num)
        getattr(bar, "finish", lambda: None)()  # call finish if it exists
        # add new events to database
        df = obsplus.events.pd._default_cat_to_df(obspy.Catalog(events=events))
        df["updated"] = update_time
        df["path"] = paths
        if len(df):
            self._write_update(self._clean_dataframe(df))
        return self

    def _clean_dataframe(self, df: pd.DataFrame):
        """ clean the dataframe by casting add dtypes """
        # fill dummy int values
        for row, dtype in COLUMN_TYPES.items():
            if dtype is int:
                df[row] = df[row].fillna(-999)
        # cast types and set index
        df = df.astype(COLUMN_TYPES)[list(COLUMN_TYPES)]
        return df.set_index("event_id")

    def _write_update(self, df: pd.DataFrame):
        """ convert updates to dataframe, then append to index table """
        # read in dataframe and cast to correct types
        assert not df.duplicated().any(), "update index has duplicate entries"

        current = self.read_index(event_id=set(df.index))
        indicies_to_update = set(current.index) & set(df.index)

        # populate index store and update metadata
        with sql_connection(self.index_path) as con:
            if indicies_to_update:  # delete rows  that will be re-entered
                _drop_rows(self._index_node, con, event_id=indicies_to_update)
            node = self._index_node
            df.to_sql(node, con, if_exists="append", index_label="event_id")
            tables = _get_tables(con)
            if self._meta_node not in tables:
                meta = self._make_meta_table()
                meta.to_sql(self._meta_node, con, if_exists="replace")
            # update timestamp
            with warnings.catch_warnings():  # ignore pandas collection warning
                warnings.simplefilter("ignore")
                dft = pd.DataFrame(time.time(), index=[0], columns=["time"])
                dft.to_sql(self._time_node, con, if_exists="replace", index=False)
        self._metadata = meta
        self._index = None

    # --- meta table

    def _read_metadata(self):
        """ return the meta table """
        self.ensure_bank_path_exists()
        with sql_connection(self.index_path) as con:
            sql = f'SELECT * FROM "{self._meta_node}";'
            return pd.read_sql(sql, con)

    # --- read events stuff

    @compose_docstring(get_events_params=get_events_parameters)
    def get_events(self, **kwargs) -> obspy.Catalog:
        """
        Read events from bank.

        Parameters
        ----------
        {get_events_params}
        """
        paths = self.bank_path + self.read_index(columns="path", **kwargs).path
        cats = (obspy.read_events(x) for x in paths)
        try:
            return reduce(add, cats)
        except TypeError:  # empty events
            return obspy.Catalog()

    def put_events(self, catalog: Union[ev.Event, ev.Catalog], update_index=True):
        """
        Put an event into the database.

        If the event_id already exists the old event will be overwritten on
        disk.

        Parameters
        ----------
        catalog
            A Catalog or Event object to put into the database.
        update_index
            Flag to indicate whether or not to update the event index after
            writing the new events. Default is True.
        """
        self.ensure_bank_path_exists(create=True)
        events = [catalog] if isinstance(catalog, ev.Event) else catalog
        # get dataframe of current event info, if they exists
        event_ids = [str(x.resource_id) for x in events]
        df = self.read_index(event_id=event_ids)
        index = set(df.index)
        for event in events:
            rid = str(event.resource_id)
            if rid in index:  # event needs to be updated
                path = self.bank_path + df.loc[rid, "path"]
                assert exists(path)
                event.write(path, self.format)
            else:  # event file does not yet exist
                path = _summarize_event(
                    event,
                    path_struct=self.path_structure,
                    name_struct=self.name_structure,
                )["path"]
                ppath = (Path(self.bank_path) / path).absolute()
                ppath.parent.mkdir(parents=True, exist_ok=True)
                event.write(str(ppath), self.format)
        if update_index:
            self.update_index()  # parse newly saved files and update index

    get_event_summary = read_index
