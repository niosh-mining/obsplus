"""
Waveframe specific utilities.
"""
import pandas as pd

from obsplus.constants import TIME_COLUMNS
from obsplus.utils import apply_funcs_to_columns, to_utc, to_datetime64


def _time_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert time columns in a pandas dataframe to UTCDateTime objects.
    """
    col_funcs = {name: to_utc for name in TIME_COLUMNS}
    return apply_funcs_to_columns(df, col_funcs)


def _time_cols_to_datetime64(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert time columns in a dataframe to numpy datetimes.
    """
    col_funcs = {name: to_datetime64 for name in TIME_COLUMNS}
    return apply_funcs_to_columns(df, col_funcs)
