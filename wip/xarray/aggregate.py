"""
Aggregation functions for xarray datastructures.
"""
import warnings
from typing import Union, Callable, Optional

import numpy as np
import xarray as xr

from obsplus.constants import xr_type
from obsplus.waveforms.xarray.accessor import ops_method
from obsplus.waveforms.xarray.utils import keep_attrs, _add_level

_AGGREGATION_FUNC = {}


def register_aggregation_func(name):
    """decorator for registering aggregation functions"""

    def _wrap(func):
        _AGGREGATION_FUNC[name] = func
        return func

    return _wrap


def _get_aggregate_from_xarray(method):
    """closure for common group aggregations"""

    def groupby_aggregate(dar: xr.DataArray, level: str, dim: str, **kwargs):
        # ignore mean of empty slice warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            group = dar.groupby(level)
            ar = getattr(group, method)(dim)
        return ar

    return groupby_aggregate


def _get_aggregate_from_ufunc(ufunc):
    """
    Closure to return aggregation function from a ufunc.

    The ufunc must accept an axis argument.
    """

    # TODO make this cleaner

    def groupby_aggregate(dar: xr.DataArray, level: str, dim: str, **kwargs):
        axis = dar.dims.index(dim)
        group = dar.groupby(level)

        def wrap_func(da, *, axis, **kwargs):
            vals = ufunc(da.values, axis=axis, **kwargs)
            values = vals.values if hasattr(vals, "values") else vals
            dims = tuple(x for x in da.dims if x != dim)
            coords = {d: da.coords[d] for d in dims}
            out = xr.DataArray(values, dims=dims, coords=coords)
            return out

        out = group.apply(wrap_func, axis=axis)
        return out

    return groupby_aggregate


@ops_method("agg")
@keep_attrs
def aggregate(
    dar: xr.DataArray,
    method: Union[str, Callable],
    level: Optional[str],
    dim: str = "seed_id",
    coord: Optional[str] = None,
    **kwargs,
):
    """
    Apply a channel aggregation over a data array using dim.

    The dimension referenced by the variable dim must be a seed id str which
    has the following format: network.station.location.channel, e.g.
    BW.RJOB..EHZ

    Parameters
    ----------
    dar
        The data array to aggregate
    method
        The method for channel aggregation, can either be a str which refers
        to a method in detex.utils.aggregate_methods dict, a callable, which
        will be passed to the xr.GroupBy.apply method, or a method supported
        by the xr.GroupBy class, like "mean" and "std"
    level
        The seed_id level over which to apply the aggregation. Supported
        options are:
            'all' - aggregate all channels in a waveforms together
            'network' - Multiplex all channels with a common network together
            'station' - Multiplex all channels with a common station together
    dim
        The dimension over which to aggregate. Will almost always be seed_id.
    coord
        A coordinate that has the dim (seed_id by default) as a subset
        that should also be truncated to reflect the aggregation. If None
        dont attempt to reconcile any coords with the aggregation level.
        For example, if a coordinate had an entry "subspace_1_UU.HDR..BHZ"
        and the aggregation was performed on station, it would be changed in
        place to "subspace_1_UU.HDR".
    Returns
    -------
    xr.DataArray
        The aggregated array, with the aggregated dimension renamed to the
        value of the variable dim
    """
    # make sure the method does exist or is a callable
    assert method in _AGGREGATION_FUNC or callable(method)
    dar, new_coord = _add_level(dar, level, dim, coord)
    assert level in dar.coords
    # get aggregation function and apply
    func = _AGGREGATION_FUNC.get(method) or _get_aggregate_from_ufunc(method)
    combined = func(dar, level, dim, **kwargs)
    if new_coord is not None:  # add the altered coord back to the data array
        combined.coords[coord] = new_coord
    combined = combined.rename({level: dim})  # rename back to seed_id
    return combined


@ops_method("bin")
@keep_attrs
def bin_array(dar: xr_type, bins: np.ndarray, raise_on_limit: bool = True) -> xr_type:
    """
    Bin the values in a datarray into specified bins.

    Parameters
    ----------
    dar
        DataArray containing numerical values
    bins
        Values of bin limits See :func: `numpy.histogram` for details
    raise_on_limit
        If True, raise a ValueError if any values in dar fall out of the bin
        limits

    Returns
    -------
    xr.DataArray
        The binned values contained in a data array where the last axis is
        switched for "bin" which represents the lower bound of the bin.
        The coord "upper_bin" represents the upper.
    """
    if raise_on_limit:  # raise if bad values encountered
        b_min, b_max = np.min(bins), np.max(bins)
        if np.any(np.logical_or(dar.values < b_min, dar.values > b_max)):
            msg = f"values in {dar} are out of bounds with {bins}"
            raise ValueError(msg)

    # perform binning
    def func(x):
        return np.histogram(x, bins=bins)[0]

    binned = np.apply_along_axis(func, -1, dar.values)
    # create/return new data array
    dims = tuple(list(dar.dims)[:-1] + ["bin"])
    coords = {item: dar.coords[item] for item in dims[:-1]}
    coords["bin"] = bins[:-1]
    coords["upper_bin"] = ("bin", bins[1:])

    return xr.DataArray(binned, coords=coords, dims=dims)


# add standard stats to aggregation functions
for method in "mean median std max min".split():
    register_aggregation_func(method)(_get_aggregate_from_xarray(method))
