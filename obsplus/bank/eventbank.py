"""
Class for interacting with events on a filesystem.
"""
import inspect
import time
from concurrent.futures import Executor
from functools import reduce, partial
from operator import add
from os.path import exists, getmtime
from pathlib import Path
from typing import Optional, Union, Sequence, Set

import numpy as np
import obspy
import obspy.core.event as ev
import pandas as pd

import obsplus
import obsplus.events.pd
from obsplus.bank.core import _Bank
from obsplus.utils.bank import (
    _IndexCache,
    sql_connection,
    _read_table,
    _get_tables,
    _drop_rows,
    _remove_base_path,
    _natify_paths,
)
from obsplus.utils.pd import (
    cast_dtypes,
    order_columns,
    _time_cols_to_ints,
    _ints_to_time_columns,
)
from obsplus.utils.events import _summarize_event, get_event_client
from obsplus.constants import (
    EVENT_PATH_STRUCTURE,
    EVENT_NAME_STRUCTURE,
    get_events_parameters,
    bar_parameter_description,
    EVENT_TYPES_OUTPUT,
    EVENT_TYPES_INPUT,
    bank_subpaths_type,
    paths_description,
)
from obsplus.events.get_events import _sanitize_circular_search, _get_ids
from obsplus.exceptions import BankDoesNotExistError
from obsplus.interfaces import ProgressBar, EventClient
from obsplus.utils import iterate
from obsplus.utils.misc import try_read_catalog, suppress_warnings
from obsplus.utils.docs import compose_docstring
from obsplus.utils.time import _dict_times_to_npdatetimes, to_datetime64

# --- define static types

# Fixed types for expected columns
# output types (ie returned from read_index)

STR_COLUMNS = {
    i
    for i, v in EVENT_TYPES_OUTPUT.items()
    if inspect.isclass(v) and issubclass(v, str)
}
INT_COLUMNS = {i for i, v in EVENT_TYPES_OUTPUT.items() if v is int}


class EventBank(_Bank):
    """
    A class to interact with a directory of event files.

    EventBank recursively reads each event file in a directory and creates
    an index to allow the files to be efficiently queried.

    Implements a superset of the :class:`~obsplus.interfaces.EventClient`
    interface.

    Parameters
    ----------
    base_path
        The path to the directory containing event files. If it does not
        exist an empty directory will be created.
    path_structure
        Defines the directory structure used by the event bank. Characters
        are separated by /, regardless of operating system. The following
        words can be used in curly braces as data specific variables:

        year, month, day, julday, hour, minute, second, event_id,
        event_id_short

        If no structure is provided it will be read from the index, if no
        index exists the default is {year}/{month}/{day}.
    name_structure
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
    cache_size
        The number of queries to store. Avoids having to read the index of
        the database multiple times for queries involving the same start and
        end times.
    executor
        An executor with the same interface as
        :py:class:`concurrent.futures.Executor, the map method of the executor
        will be used for reading files and updating indices.

    Attributes
    ----------
    name_structure

    Examples
    --------
    >>> # --- Create an `EventBank` from a path to a directory with quakeml files.
    >>> import obsplus
    >>> event_path = obsplus.copy_dataset('default_test').event_path
    >>> # init an EventBank and index the event files.
    >>> ebank = obsplus.EventBank(event_path).update_index()

    >>> # --- Retrieve catalog objects from the bank.
    >>> cat = ebank.get_events(minmagnitude=4.3, minlatitude=40.12)
    >>> print(cat)
    1 Event(s) in Catalog:...

    >>> # --- Put event files bank into the bank.
    >>> # get an event from another dataset, keep track of its id
    >>> ds = obsplus.load_dataset('bingham_test')
    >>> new_events = ds.event_client.get_events(limit=1)
    >>> new_event_id = str(new_events[0].resource_id)
    >>> # put the event into the EventBank
    >>> _ = ebank.put_events(new_events)
    >>> print(ebank.get_events(eventid=new_event_id))
    1 Event(s) in Catalog:...

    >>> # --- Read the index used by EventBank as a DataFrame.
    >>> df = ebank.read_index()
    >>> assert len(df) == 4, 'there should now be 4 events in the bank.'
    """

    namespace = "/events"
    index_name = ".index.db"  # name of index file
    _min_files_for_bar = 50
    _dtypes_output = EVENT_TYPES_OUTPUT
    _dtypes_input = EVENT_TYPES_INPUT
    _max_events_in_memory = 2000

    def __init__(
        self,
        base_path: Union[str, Path, "EventBank"] = ".",
        path_structure: Optional[str] = None,
        name_structure: Optional[str] = None,
        cache_size: int = 5,
        format="quakeml",
        ext=".xml",
        executor: Optional[Executor] = None,
    ):
        """ Initialize an instance. """
        if isinstance(base_path, EventBank):
            self.__dict__.update(base_path.__dict__)
            return
        self.bank_path = Path(base_path).absolute()
        self._index = None
        self.format = format
        self.ext = ext
        # get waveforms structure based on structures of path and filename
        ps = (
            path_structure
            if path_structure is not None
            else (self._path_structure or EVENT_PATH_STRUCTURE)
        )
        self.path_structure = ps
        ns = name_structure or self._name_structure or EVENT_NAME_STRUCTURE
        self.name_structure = ns
        self.executor = executor
        # initialize cache
        self._index_cache = _IndexCache(self, cache_size=cache_size)
        # enforce min version and warn on newer
        self._enforce_min_version()
        self._warn_on_newer_version()

    @property
    def last_updated_timestamp(self):
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
        # Make sure all times are numpy datetime64
        kwargs = _dict_times_to_npdatetimes(kwargs)
        # a simple switch to prevent infinite recursion
        allow_update = kwargs.pop("_allow_update", True)
        # Circular search requires work to be done on the dataframe - we need
        # to get the whole dataframe then calculate the distances and search in
        # that
        circular_kwargs, kwargs = _sanitize_circular_search(**kwargs)
        with sql_connection(self.index_path) as con:
            try:
                df = _read_table(self._index_node, con, **kwargs)
            except pd.io.sql.DatabaseError:
                # if this database has never been updated, update now
                if allow_update and self.last_updated_timestamp < 1:
                    self.update_index()
                    return self.read_index(_allow_update=False, **kwargs)
                # else return empty index
                df = pd.DataFrame(columns=list(EVENT_TYPES_OUTPUT))
        df = _ints_to_time_columns(df, columns=INT_COLUMNS).pipe(
            self._prepare_dataframe, dtypes=EVENT_TYPES_OUTPUT
        )
        if len(circular_kwargs) >= 3:
            # Requires at least latitude, longitude and min or max radius
            circular_ids = _get_ids(df, circular_kwargs)
            df = df[df.event_id.isin(circular_ids)]
        return df

    @compose_docstring(
        bar_description=bar_parameter_description,
        subpaths_description=paths_description,
    )
    def update_index(
        self,
        bar: Optional[ProgressBar] = None,
        paths: Optional[bank_subpaths_type] = None,
    ) -> "EventBank":
        """
        Iterate files in bank and add any modified since last update to index.

        Parameters
        ----------
        {bar_parameter_description}
        {paths_description}
        """
        bank_path = str(self.bank_path)

        self._enforce_min_version()  # delete index if schema has changed
        # create iterator  and lists for storing output
        update_time = time.time()
        # create an iterator which yields files to update and updates bar
        file_yielder = self._unindexed_iterator(paths=paths)
        update_file_feeder = self._measure_iterator(file_yielder, bar)
        new_func = partial(
            self._get_cat_update_time_path, bank_path=bank_path, format=self.format
        )
        # create iterator, loop over it in chunks until it is exhausted
        iterator = self._map(new_func, update_file_feeder)
        events_remain = True
        while events_remain:
            events_remain = self._index_from_iterable(iterator, update_time)
        return self

    @staticmethod
    def _get_cat_update_time_path(path, bank_path, format):
        """ Function to yield events, update_time and paths. """
        # NOTE: This function must be static to avoid pickling the attached
        # executor. (see #158).
        cat = try_read_catalog(path, format=format)
        update_time = getmtime(path)
        path = path.replace(bank_path, "")
        return cat, update_time, path

    def _index_from_iterable(self, iterable, update_time):
        """Iterate over an event iterable and dump to database."""
        events, update_times, paths = [], [], []
        max_mem = self._max_events_in_memory  # this avoids the MRO each loop
        events_remain = False

        for cat, mtime, path in iterable:
            if cat is None:
                continue
            for event in cat:
                events.append(event)
                update_times.append(mtime)
                paths.append(path)
            if len(events) >= max_mem:  # max limit exceeded, dump to db
                events_remain = True
                break
        # add new events to database
        df = obsplus.events.pd._default_cat_to_df(events)
        df["updated"] = to_datetime64(update_times)
        df["path"] = _remove_base_path(pd.Series(paths, dtype=object))
        if len(df):
            df = _time_cols_to_ints(df)
            df_to_write = self._prepare_dataframe(df, EVENT_TYPES_INPUT)
            self._write_update(df_to_write, update_time)
        return events_remain

    def _prepare_dataframe(self, df: pd.DataFrame, dtypes: dict):
        """
        Fill missing values and casting data types.
        """
        # replace "None" with empty string for str columns
        str_cols = STR_COLUMNS & set(df.columns)
        df.loc[:, str_cols] = df.loc[:, str_cols].replace(["None"], [""])
        # get expected datatypes
        assert set(INT_COLUMNS | STR_COLUMNS).issubset(set(dtypes))
        intersection = set(dtypes) & set(df.columns)
        dtype = {i: dtypes[i] for i in dtypes if i in intersection}
        # order columns, set types, reset index
        out = (
            df.loc[:, ~df.columns.duplicated()]
            .pipe(cast_dtypes, dtype=dtype)  # drop dup. columns
            .pipe(order_columns, required_columns=list(dtype), drop_columns=True)
            .reset_index(drop=True)
        )
        # convert low ints back to NaT

        return out

    def _write_update(self, df: pd.DataFrame, update_time=None):
        """ convert updates to dataframe, then append to index table """
        # read in dataframe and cast to correct types
        assert not df.duplicated().any(), "update index has duplicate entries"

        # set both dfs to use index of event_id
        df = df.set_index("event_id")
        # get current events, but dont allow it to update again
        current = self.read_index(event_id=set(df.index), _allow_update=False)
        indicies_to_update = set(current["event_id"]) & set(df.index)
        # populate index store and update metadata
        with sql_connection(self.index_path) as con:
            if indicies_to_update:  # delete rows that will be re-entered
                _drop_rows(self._index_node, con, event_id=indicies_to_update)
            node = self._index_node
            df.to_sql(node, con, if_exists="append", index_label="event_id")
            tables = _get_tables(con)
            if self._meta_node not in tables:
                meta = self._make_meta_table()
                meta.to_sql(self._meta_node, con, if_exists="replace")
            # update timestamp
            with suppress_warnings():  # ignore pandas collection warning
                timestamp = update_time or time.time()
                dft = pd.DataFrame(timestamp, index=[0], columns=["time"])
                dft.to_sql(self._time_node, con, if_exists="replace", index=False)
        self._metadata = meta
        self._index = None

    # --- meta table

    def _read_metadata(self):
        """ return the meta table """
        self.ensure_bank_path_exists()
        with sql_connection(self.index_path) as con:
            sql = f'SELECT * FROM "{self._meta_node}";'
            out = pd.read_sql(sql, con)
        return out

    # --- read events stuff

    @compose_docstring(get_events_params=get_events_parameters)
    def get_events(self, **kwargs) -> obspy.Catalog:
        """
        Read events from bank.

        Parameters
        ----------
        {get_events_params}
        """
        ind = self.read_index(**kwargs)
        eids = ind["event_id"]
        file_paths = ind["path"]
        paths = str(self.bank_path) + _natify_paths(file_paths)
        paths.drop_duplicates(inplace=True)
        read_func = partial(try_read_catalog, format=self.format)
        # Divide work evenly between workers, with a min chunksize of 1.
        chunksize = len(paths) // self._max_workers or 1
        map_kwargs = dict(chunksize=chunksize)
        try:
            mapped_values = self._map(read_func, paths.values, **map_kwargs)
            cat = reduce(add, mapped_values)
        except TypeError:  # empty events
            cat = obspy.Catalog()
        # Make sure only the events of interest are included
        return obspy.Catalog([eve for eve in cat if eve.resource_id.id in eids.values])

    def ids_in_bank(self, event_id: Union[str, Sequence[str]]) -> Set[str]:
        """
        Determine if one or more event_ids are used by the bank.

        This function is faster than reading the entire index into memory to
        perform a similar check.

        Parameters
        ----------
        event_id
            A single event id or sequence of event ids.

        Returns
        -------
            A set of event_ids which are also found in the bank.
        """
        eids = self.read_index(columns="event_id").values
        unique = set(np.unique(eids))
        return unique & {str(x) for x in iterate(event_id)}

    @compose_docstring(bar_parameter_description=bar_parameter_description)
    def put_events(
        self,
        events: Union[ev.Event, ev.Catalog, EventClient],
        update_index: bool = True,
        bar: Optional[ProgressBar] = None,
    ) -> "EventBank":
        """
        Put events into the EventBank.

        If the event_id already exists the old event will be overwritten on
        disk.

        Parameters
        ----------
        events
            An objects which contains event data (e.g. Catalog, Event,
            EventBank, etc.)
        update_index
            Flag to indicate whether or not to update the event index after
            writing the new events. Note: Only events added through this
            method call will get indexed. Default is True.
        {bar_parameter_description}

        Notes
        -----
        If any of the events do not have an extractable reference time a
        ``ValueError`` will be raised.
        """
        self.ensure_bank_path_exists(create=True)
        # get catalog from any event client
        events = get_event_client(events).get_events()
        # get index only with needed resource_ids
        event_ids = [str(x.resource_id) for x in events]
        df = self.read_index(event_id=event_ids).set_index("event_id")
        # create an iterator and apply over potential pool
        event_feeder = self._measure_iterator(events, bar)
        new_func = partial(
            self._put_event,
            index=df,
            bank_path=self.bank_path,
            path_structure=self.path_structure,
            name_structure=self.name_structure,
            format=self.format,
        )
        paths = list(self._map(new_func, event_feeder))
        if update_index:  # parse newly saved files and update index
            self.update_index(paths=paths)
        return self

    @staticmethod
    def _put_event(event, index, bank_path, path_structure, name_structure, format):
        """
        Get a single event's path, save to db, return path.
        """
        # NOTE: This function must be static to avoid pickling the attached
        # executor (see #158)
        bank_path = str(bank_path)
        df = index
        rid = str(event.resource_id)
        if rid in df.index:  # event needs to be updated
            path = df.loc[rid, "path"]
            save_path = bank_path + path
            assert exists(save_path)
        else:  # event file does not yet exist
            path = _summarize_event(
                event, path_struct=path_structure, name_struct=name_structure
            )["path"]
            save_path = (Path(bank_path) / path).absolute()
        path = Path(save_path)
        path.parent.mkdir(exist_ok=True, parents=True)
        event.write(str(path), format=format)
        return path

    get_event_summary = read_index
