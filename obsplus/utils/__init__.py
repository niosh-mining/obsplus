"""
General utilities for obsplus.
"""
# from obsplus.utils.pd import get_seed_id_series
# from obsplus.utils.time import to_utc

# make a dict of functions for reading waveforms

from obsplus.utils.pd import get_seed_id_series

# added for compatibility
get_nslc_series = get_seed_id_series
