"""
Methods for saving/reading xarray streams from disk.
"""
import os
from typing import Optional, Union, Any

import cloudpickle
import numpy as np
import xarray as xr

from obsplus.waveforms.xarray import ops_method


@ops_method("to_netcdf")
def array2netcdf(
    dar: xr.DataArray, path: Optional[str] = None, **kwargs
) -> Optional[bytes]:
    """
    Write a data array to netcdf format.

    A special form of this function is provided to serialize any python
    objects in the attrs dict using cloud pickle.

    Parameters
    ----------
    dar
        Data Array containing continuous data
    path
        The path to save file to disk, if None, a byte rep. is returned
    kwargs
        Keyword args passed to to_netcdf function

    Notes
    ---------
    See :function: ~xr.DataArray.to_netcdf for supported kwargs
    """
    byte_str = cloudpickle.dumps(dar.attrs)
    old_attrs = dar.attrs
    dar.attrs = {"attrs": np.frombuffer(byte_str, dtype=np.int8)}
    out = dar.to_netcdf(path=path, **kwargs)
    dar.attrs = old_attrs
    return out


def netcdf2array(filepath_or_obj: Union[str, bytes]) -> xr.DataArray:
    """
    Read an xarray object from a file or bytes.

    A special form of this function is provided to read serialized python
    objects in the attrs dict using cloud pickle.

    Parameters
    ----------
    filepath_or_obj
        The path to the file or a bytes object

    Notes
    ---------
    See :function: ~xr.DataArray.to_netcdf for supported kwargs
    """
    dar = xr.open_dataarray(filepath_or_obj)
    dar.load()  # load for now, if "big data" happens reconsider
    byts = dar.attrs["attrs"].tobytes()
    dar.attrs = cloudpickle.loads(byts)
    return dar


@ops_method("to_pickle")
def write_pickle(obj: Any, file_name: Optional[str] = None) -> Optional[bytes]:
    """
    Serialize an object to bytes using cloudpickle.

    Parameters
    ----------
    obj
        Any object
    file_name
        The path to save to. If None, return a waveforms of bytes

    Returns
    -------
    bytes
        If file_name is None return a waveforms of bytes (in memory file)
    """
    if file_name:
        with open(file_name, "wb") as fi:
            cloudpickle.dump(obj, fi, protocol=-1)
    else:
        return cloudpickle.dumps(obj, protocol=-1)


def read_pickle(path_or_bytes: Union[str, bytes]):
    """
    Read a pickled object from a byte string
    Parameters
    ----------
    path_or_bytes

    Returns
    -------
    Any
        Whatever was pickled
    """
    if isinstance(path_or_bytes, bytes):
        return cloudpickle.loads(path_or_bytes)
    elif isinstance(path_or_bytes, str):
        assert os.path.exists(path_or_bytes)
        with open(path_or_bytes, "rb") as fi:
            return cloudpickle.load(fi)
