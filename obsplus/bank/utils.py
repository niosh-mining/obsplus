"""
Utils for banks
"""
import contextlib
import itertools
import os
import re
import sqlite3
import time
import warnings
from functools import singledispatch
from os.path import join
from typing import Optional, Sequence, Union

import obspy
import obspy.core.event as ev
import pandas as pd
from obspy import Inventory
from tables.exceptions import ClosedNodeError

from obsplus.constants import (
    NSLC,
    WAVEFORM_STRUCTURE,
    EVENT_PATH_STRUCTURE,
    WAVEFORM_NAME_STRUCTURE,
    EVENT_NAME_STRUCTURE,
    SMALLDT64,
    LARGEDT64,
    MININT64,
)
from obsplus.utils import (
    _get_event_origin_time,
    READ_DICT,
    dict_times_to_ns,
    to_datetime64,
)
from .mseed import summarize_mseed

# functions for summarizing the various formats
summarizing_functions = dict(mseed=summarize_mseed)

# extensions
WAVEFORM_EXT = ".mseed"
EVENT_EXT = ".xml"
STATION_EXT = ".xml"


# name structures


def _get_time_values(time1, time2=None):
    """ get the time values from a UTCDateTime object or two """
    tvals = "year month day hour minute second microsecond".split()
    utc1 = time1
    split = re.split("-|:|T|[.]", str(utc1).replace("Z", ""))
    assert len(tvals) == len(split)
    out = {key: val for key, val in zip(tvals, split)}
    out["julday"] = "%03d" % utc1.julday
    out["starttime"] = utc1.timestamp
    if time2:
        out["endtime"] = time2.timestamp
    out["time"] = str(utc1).replace(":", "-").split(".")[0]
    return out


def _get_path(info, path, name, path_struct, name_strcut):
    """ return a dict with path, and file name """
    if path is None:  # if the path needs to be created
        ext = info.get("ext", "")
        # get name
        fname = name or name_strcut.format_map(info)
        fname = fname if fname.endswith(ext) else fname + ext  # add ext
        # get structure
        psplit = path_struct.format_map(info).split("/")
        path = join(*psplit, fname)
        out_name = fname
    else:  # if the path is already known
        out_name = os.path.basename(path)
    return dict(path=path, filename=out_name)


def summarize_generic_stream(path, format=None):
    st = _try_read_stream(path, format=format) or _try_read_stream(path) or []
    out = []
    for tr in st:
        summary = {
            "starttime": tr.stats.starttime._ns,
            "endtime": tr.stats.endtime._ns,
            "sampling_period": int(tr.stats.delta * 1_000_000_000),
            "path": path,
        }
        summary.update(dict((x, c) for x, c in zip(NSLC, tr.id.split("."))))
        out.append(summary)
    return out


def _summarize_wave_file(path, format, summarizer=None):
    """
    Summarize waveform files for indexing.

    Note: this function is a bit ugly, but it gets called *a lot* so be sure
    to profile before refactoring.
    """
    if summarizer is not None:
        try:
            return summarizer(path)
        except Exception:
            pass
    return summarize_generic_stream(path, format)


def _summarize_trace(
    trace: obspy.Trace,
    path: Optional[str] = None,
    name: Optional[str] = None,
    path_struct: Optional[str] = None,
    name_struct: Optional[str] = None,
) -> dict:
    """
    Function to extract info from traces for indexing.

    Parameters
    ----------
    trace
        The trace object
    path
        Other Parameters to the file
    name
        Name of the file
    path_struct
        directory structure to create
    name_struct
    """
    assert hasattr(trace, "stats"), "only a trace object is accepted"
    out = {"seedid": trace.id, "ext": WAVEFORM_EXT}
    t1, t2 = trace.stats.starttime, trace.stats.endtime
    out.update(_get_time_values(t1, t2))
    out.update(dict((x, c) for x, c in zip(NSLC, trace.id.split("."))))

    path_struct = path_struct or WAVEFORM_STRUCTURE
    name_struct = name_struct or WAVEFORM_NAME_STRUCTURE

    out.update(_get_path(out, path, name, path_struct, name_struct))
    return out


def _summarize_event(
    event: ev.Event,
    path: Optional[str] = None,
    name: Optional[str] = None,
    path_struct: Optional[str] = None,
    name_struct: Optional[str] = None,
) -> dict:
    """
    Function to extract info from events for indexing.

    Parameters
    ----------
    event
        The event object
    path
        Other Parameters to the file
    name
        Name of the file
    path_struct
        directory structure to create
    name_struct
    """
    res_id = str(event.resource_id)
    out = {
        "ext": EVENT_EXT,
        "event_id": res_id,
        "event_id_short": res_id[-5:],
        "event_id_end": res_id.split("/")[-1],
    }
    t1 = _get_event_origin_time(event)
    out.update(_get_time_values(t1))
    path_struct = path_struct or EVENT_PATH_STRUCTURE
    name_struct = name_struct or EVENT_NAME_STRUCTURE

    out.update(_get_path(out, path, name, path_struct, name_struct))
    return out


class _IndexCache:
    """ A simple class for caching indexes """

    def __init__(self, bank, cache_size=5):
        self.max_size = cache_size
        self.bank = bank
        self.cache = pd.DataFrame(
            index=range(cache_size), columns="t1 t2 kwargs cindex".split()
        )
        self.next_index = itertools.cycle(self.cache.index)

    def __call__(self, starttime, endtime, buffer, **kwargs):
        """ get start and end times, perform in kernel lookup """
        # get defaults if starttime or endtime is none
        starttime = to_datetime64(starttime or SMALLDT64)
        endtime = to_datetime64(endtime or LARGEDT64)
        # find out if the query falls within one cached times
        con1 = self.cache.t1 <= starttime
        con2 = self.cache.t2 >= endtime
        con3 = self.cache.kwargs == self._kwargs_to_str(kwargs)
        cached_index = self.cache[con1 & con2 & con3]
        if not len(cached_index):  # query is not cached get it from hdf5 file
            where = get_kernel_query(int(starttime), int(endtime), int(buffer))
            raw_index = self._get_index(where, **kwargs)
            # replace "None" with None
            ic = self.bank.index_str
            raw_index.loc[:, ic] = raw_index.loc[:, ic].replace(["None"], [None])
            # convert data types used by bank back to those seen by user
            index = raw_index.astype(dict(self.bank._dtypes_output))
            self._set_cache(index, starttime, endtime, kwargs)
        else:
            index = cached_index.iloc[0]["cindex"]
        # trim down index
        con1 = index["starttime"] >= (endtime + buffer)
        con2 = index["endtime"] <= (starttime - buffer)
        return index[~(con1 | con2)]

    def _set_cache(self, index, starttime, endtime, kwargs):
        """ cache the current index """
        ser = pd.Series(
            {
                "t1": starttime,
                "t2": endtime,
                "cindex": index,
                "kwargs": self._kwargs_to_str(kwargs),
            }
        )
        self.cache.loc[next(self.next_index)] = ser

    def _kwargs_to_str(self, kwargs):
        """ convert kwargs to a string """
        keys = sorted(list(kwargs.keys()))
        ou = str([(item, kwargs[item]) for item in keys])
        return ou

    def _get_index(self, where, fail_counts=0, **kwargs):
        """ read the hdf5 file """
        try:
            return pd.read_hdf(
                self.bank.index_path, self.bank._index_node, where=where, **kwargs
            )

        except (ClosedNodeError, Exception) as e:
            # Sometimes in concurrent updates the nodes need time to open/close
            if fail_counts > 10:
                raise e
            # Wait a bit and try again (up to 10 times)
            time.sleep(0.1)
            return self._get_index(where, fail_counts=fail_counts + 1, **kwargs)

    def clear_cache(self):
        """ removes all cached dataframes. """
        self.cache = pd.DataFrame(
            index=range(self.max_size), columns="t1 t2 kwargs cindex".split()
        )


@contextlib.contextmanager
def sql_connection(path, **kwargs):
    con = sqlite3.connect(path, **kwargs)
    with con:
        yield con


def get_kernel_query(starttime: int, endtime: int, buffer: int):
    """" get the HDF5 kernel query parameters (this is necessary because
    hdf5 doesnt accept invert conditions for some reason. A slight buffer
    is applied to the ranges to make sure no edge files are left out"""
    t1 = starttime - buffer
    t2 = endtime + buffer
    con = (
        f"(starttime>{t1:d} & starttime<{t2:d}) | "
        f"((endtime>{t1:d} & endtime<{t2:d}) | "
        f"(starttime<{t1:d} & endtime>{t2:d}))"
    )
    return con


# --- SQL stuff


def _str_of_params(value):
    """
    Make sure a list of params is returned.

    This allows user to specify a single parameter, a list, set, nparray, etc.
    to match on.
    """
    if isinstance(value, str):
        return value
    else:
        # try to coerce in a list of str
        try:
            return [str(x) for x in value]
        except TypeError:  # else fallback to str repr
            return str(value)


def _make_wheres(queries):
    """ Create the where queries, join with AND clauses """

    def _rename_keys(kwargs):
        """ re-word some keys to make automatic sql generation easier"""
        if "eventid" in kwargs:
            kwargs["event_id"] = kwargs["eventid"]
            kwargs.pop("eventid")
        if "event_id" in kwargs:
            kwargs["event_id"] = _str_of_params(kwargs["event_id"])
        if "event_description" in kwargs:
            kwargs["event_description"] = _str_of_params(kwargs["event_description"])
        if "endtime" in kwargs:
            kwargs["maxtime"] = kwargs.pop("endtime")
        if "starttime" in kwargs:
            kwargs["mintime"] = kwargs.pop("starttime")
        return kwargs

    def _handle_nat(kwargs):
        """ add a mintime that will exclude NaT values if endtime is used """
        if "maxtime" in kwargs and "mintime" not in kwargs:
            kwargs["mintime"] = MININT64 + 1
        return kwargs

    def _build_query(kwargs):
        """ iterate each key/value and build query """
        out = []
        for key, val in kwargs.items():
            # deal with simple min/max
            if key.startswith("min"):
                out.append(f"{key.replace('min', '')} > {val}")
            elif key.startswith("max"):
                out.append(f"{key.replace('max', '')} < {val}")
            # deal with equals or ins
            elif isinstance(val, Sequence):
                if isinstance(val, str):
                    val = [val]
                tup = str(tuple(val)).replace(",)", ")")  # no trailing coma
                out.append(f"{key} IN {tup}")
            else:
                out.append(f"{key} = {val}")
        return " AND ".join(out).replace("'", '"')

    kwargs = dict_times_to_ns(queries)
    kwargs = _rename_keys(kwargs)
    kwargs = _handle_nat(kwargs)
    return _build_query(kwargs)


def _make_sql_command(cmd, table_name, columns=None, **kwargs) -> str:
    """ build a sql command """
    # get columns
    if columns:
        col = [columns] if isinstance(columns, str) else columns
        col += ["event_id"]  # event_id is used as index
        columns = ", ".join(col)
    elif cmd.upper() == "DELETE":
        columns = ""
    else:
        columns = "*"
    limit = kwargs.pop("limit", None)
    wheres = _make_wheres(kwargs)
    sql = f'{cmd.upper()} {columns} FROM "{table_name}"'
    if wheres:
        sql += f" WHERE {wheres}"
    if limit:
        sql += f" LIMIT {limit}"
    return sql + ";"


def _read_table(table_name, con, columns=None, **kwargs) -> pd.DataFrame:
    """
    Read a SQLite table.

    Parameters
    ----------
    table_name
    con
    columns

    Returns
    -------

    """
    # first ensure all times are ns (as ints)
    sql = _make_sql_command("select", table_name, columns=columns, **kwargs)
    # replace "None" with None
    return pd.read_sql(sql, con)


def _get_tables(con):
    """ Return a list of table in sqlite database """
    out = con.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return set(out)


def _drop_rows(table_name, con, columns=None, **kwargs):
    """ Drop indicies in table """
    sql = _make_sql_command("delete", table_name, columns=columns, **kwargs)
    con.execute(sql)


def _try_read_stream(stream_path, format=None, **kwargs):
    """" Try to read a waveforms from file, if raises return None """
    read = READ_DICT.get(format, obspy.read)
    stt = None
    try:
        stt = read(stream_path, **kwargs)
    except Exception:
        try:
            stt = obspy.read(stream_path, **kwargs)
        except Exception:
            warnings.warn("obspy failed to read %s" % stream_path, UserWarning)
        else:
            msg = f"{stream_path} was read but is not of format {format}"
            warnings.warn(msg, UserWarning)
    finally:
        return stt if stt else None


@singledispatch
def get_inventory(inventory: Union[str, Inventory]):
    """
    Get an stations from stations parameter if path or stations else
    return None

    Parameters
    ----------
    inventory : str, obspy.Inventory, or None

    Returns
    -------
    obspy.Inventory or None
    """
    assert isinstance(inventory, Inventory) or inventory is None
    return inventory


@get_inventory.register(str)
def _get_inv_str(inventory):
    """ if str is provided """
    return obspy.read_inventory(inventory)
