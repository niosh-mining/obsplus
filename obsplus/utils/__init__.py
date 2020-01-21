"""
General utilities for obsplus.
"""
# from obsplus.utils.pd import get_seed_id_series
# from obsplus.utils.time import to_utc

# make a dict of functions for reading waveforms

from obsplus.utils.misc import iterate, yield_obj_parent_attr
from obsplus.utils.pd import get_seed_id_series

# added for compatibility
get_nslc_series = get_seed_id_series
