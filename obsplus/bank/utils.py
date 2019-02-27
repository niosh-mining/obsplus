"""
Utils for banks
"""
import contextlib
import itertools
import os
import re
import sqlite3
import warnings
from functools import singledispatch
from os.path import join
from typing import Optional, Sequence, Union

import obspy
import obspy.core.event as ev
import numpy as np
import pandas as pd

from obsplus.constants import (
    NSLC,
    WAVEFORM_STRUCTURE,
    EVENT_PATH_STRUCTURE,
    WAVEFORM_NAME_STRUCTURE,
    EVENT_NAME_STRUCTURE,
)
from obsplus.utils import _get_event_origin_time, READ_DICT
from obspy import Inventory

from .mseed import summarize_mseed

# --- sensible defaults

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


def _summarize_wave_file(path, format):
    if format == "mseed":
        try:
            return summarize_mseed(path)
        except Exception:
            pass
    # specialized mseed function failed
    out_list = []
    st = _try_read_stream(path, format=format) or _try_read_stream(path) or []
    for tr in st:
        out = {
            "starttime": tr.stats.starttime.timestamp,
            "endtime": tr.stats.endtime.timestamp,
            "path": path,
        }
        out.update(dict((x, c) for x, c in zip(NSLC, tr.id.split("."))))
        out_list.append(out)
    return out_list


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
    out = {"ext": EVENT_EXT, "event_id": res_id, "event_id_short": res_id[-5:]}
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
        starttime = obspy.UTCDateTime(starttime or 0.0).timestamp
        endtime = obspy.UTCDateTime(endtime or 9_000_000_000).timestamp
        # find out if the query falls within one cached times
        con1 = self.cache.t1 <= starttime
        con2 = self.cache.t2 >= endtime
        con3 = self.cache.kwargs == self._kwargs_to_str(kwargs)
        cached_index = self.cache[con1 & con2 & con3]
        if not len(cached_index):  # query is not cached get it from cache
            # get expected dtypes
            strs = {x: str for x in self.bank.index_str}
            floats = {x: float for x in self.bank.index_float}
            dtypes = {**strs, **floats}
            where = get_kernel_query(starttime, endtime, buffer=buffer)
            index = self._get_index(where, **kwargs)
            # replace "None" with None
            ic = self.bank.index_str
            index.loc[:, ic] = index.loc[:, ic].replace(["None"], [None])
            self._set_cache(index.astype(dtypes), starttime, endtime, kwargs)
        else:
            index = cached_index.iloc[0]["cindex"]
        # trim down index
        con1 = index.starttime >= (endtime + buffer)
        con2 = index.endtime <= (starttime - buffer)
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

    def _get_index(self, where, **kwargs):
        """ read the hdf5 file """
        return pd.read_hdf(
            self.bank.index_path, self.bank._index_node, where=where, **kwargs
        )

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


def get_kernel_query(starttime: float, endtime: float, buffer: float):
    """" get the HDF5 kernel query parameters (this is necessary because
    hdf5 doesnt accept invert conditions for some reason. A slight buffer
    is applied to the ranges to make sure no edge files are left out"""
    t1 = obspy.UTCDateTime(starttime).timestamp - buffer
    t2 = obspy.UTCDateTime(endtime).timestamp + buffer
    con = (
        f"(starttime>{t1:f} & starttime<{t2:f}) | "
        f"((endtime>{t1:f} & endtime<{t2:f}) | "
        f"(starttime<{t1:f} & endtime>{t2:f}))"
    )
    return con


# --- SQL stuff


def _make_wheres(queries):
    """ Create the where queries, join with AND clauses """
    kwargs = dict(queries)
    out = []

    if "eventid" in kwargs:
        kwargs["event_id"] = kwargs["eventid"]
        kwargs.pop("eventid")
    if "event_id" in kwargs:
        val = kwargs.pop("event_id")
        seq_types = (Sequence, set, np.ndarray)
        if isinstance(val, seq_types) and not isinstance(val, str):
            kwargs["event_id"] = [str(x) for x in val]
        else:
            kwargs["event_id"] = str(val)
    if "endtime" in kwargs:
        kwargs["maxtime"] = kwargs["endtime"]
        kwargs.pop("endtime")
    if "starttime" in kwargs:
        kwargs["mintime"] = kwargs["starttime"]
        kwargs.pop("starttime")
    if "mintime" in kwargs:
        kwargs["mintime"] = obspy.UTCDateTime(kwargs["mintime"]).timestamp
    if "maxtime" in kwargs:
        kwargs["maxtime"] = obspy.UTCDateTime(kwargs["maxtime"]).timestamp

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
        if stt is not None and len(stt):
            return stt
    return None


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
