from typing import Optional

import numpy as np
import xarray as xr

from obsplus.waveforms.xarray.accessor import ops_method
from obsplus.waveforms.xarray.utils import keep_attrs


def _get_frequencies(dar, dim, dim_len=None):
    """ get a list of frequencies each complex point represents, return in
    dict for coords input"""
    dim_len = dim_len or len(dar[dim])  # the req_len of the dimension
    try:  # if metadata are still attached (they come off easy)
        sample_spacing = 1.0 / dar.attrs.get("sampling_rate")
    except (KeyError, TypeError):
        sample_spacing = float(dar[dim][1] - dar[dim][0])
    finally:
        freqs = np.fft.rfftfreq(dim_len, sample_spacing)
        return freqs


@ops_method("irfft")
@keep_attrs
def array_irfft(dar: xr.DataArray, old_dim: str = "frequency", new_dim: str = "time"):
    """
    Perform an inverse fft on the array that has a frequency dim, converting
    it to "time"

    Parameters
    ----------
    dar : xr.DataArray
        An array created by array_fft
    old_dim : set
        The dimensions expected to perform the fft along
    new_dim : str
        The name of the output dimension
    """
    dim_coord, dims, coords = _get_new_dims(dar, old_dim, new_dim, _get_times)
    fft_ar = np.real(np.fft.irfft(dar.values, axis=dim_coord))
    ar = xr.DataArray(fft_ar, dims=dims, coords=coords)
    return ar


def _get_times(dar, dim, dim_len=None):
    """ get a list of times returned by the ifft """
    try:  # if metadata are still attached (they come off easy)
        sr = 1.0 / dar.attrs["sampling_rate"]
    except (KeyError, TypeError):
        sr = float(np.round(dar.frequency.max()))
    finally:  # account for rffts being used
        time_len = (len(dar[dim]) - 1) * 2  # since only rfft is used here
        duration = sr * (time_len - 1)
        times = np.linspace(0, duration, time_len)
        return times


def _get_new_dims(dar, old_dim, output_dim, out_coord_func, dim_len=None):
    """
    get the new dimensions for the array, return list of new dimensions
    and the name of the old dimension
    """
    dims = list(dar.dims)  # TODO find a better way to do this...
    dim_coord = dims.index(old_dim)
    d_coords = {d: dar[d].values for d in dims}
    # swap out dimension that is to be replaced
    d_coords[old_dim] = out_coord_func(dar, old_dim, dim_len=dim_len)
    new_dims = [output_dim if d == old_dim else d for d in dims]
    coords = {d_new: d_coords[d_old] for d_old, d_new in zip(dims, new_dims)}
    # add non-dimensional coords
    non_doms = list(set(dar.coords) - set(d_coords))
    for non_dom in non_doms:
        coords[non_dom] = dar.coords[non_dom]
    return dim_coord, new_dims, coords


@ops_method("rfft")
@keep_attrs
def array_rfft(
    dar: xr.DataArray,
    old_dim: str = "time",
    new_dim: str = "frequency",
    required_len: Optional[int] = None,
) -> xr.DataArray:
    """
    Perform an fft on the array to get frequency domain representation

    Parameters
    ----------
    dar : xr.DataArray
        An array created by stream2array or stream2dataset
    old_dim : set
        The dimensions expected to perform the fft along
    new_dim : str
        The name of the output dimension
    required_len : int
        The required req_len for the output
    """
    # determine required req_len
    required_len = required_len or len(dar[old_dim])
    # get the dimension the fft will be performed along, get new dimensions
    ax, dims, coords = _get_new_dims(
        dar, old_dim, new_dim, _get_frequencies, dim_len=required_len
    )
    # perform fft
    fft_ar = np.fft.rfft(dar.values, required_len, axis=ax)
    ar = xr.DataArray(fft_ar, dims=dims, coords=coords)
    return ar
