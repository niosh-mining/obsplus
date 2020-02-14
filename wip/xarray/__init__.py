"""
Functionality for working with Xarray for seismology data.
"""

from obsplus.waveforms.xarray.accessor import ops_method, OPS_METHODS

from obsplus.waveforms.xarray.utils import keep_attrs, _add_level
from obsplus.waveforms.xarray.io import netcdf2array
from obsplus.utils import get_seed_id_series


# load in all aggregation plugins
from obsplus.waveforms.xarray.aggregate import (
    register_aggregation_func,
    _get_aggregate_from_xarray,
)

# this ensures the ops methods in this module are loaded
from obsplus.waveforms.xarray.signal import array_rfft
