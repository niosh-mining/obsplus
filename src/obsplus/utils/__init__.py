"""
General utilities for obsplus.
"""
from obsplus.utils.misc import iterate, yield_obj_parent_attr
from obsplus.utils.pd import get_seed_id_series
from obsplus.utils.time import to_datetime64, to_timedelta64, to_utc, get_reference_time
from obsplus.utils.geodetics import SpatialCalculator

# added for compatibility
get_nslc_series = get_seed_id_series
