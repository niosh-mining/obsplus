"""
This module contains an experimental functions for converting coordinate systems. The interfaces for these functions
likely will not change, but the functions may be moved.
"""

from copy import copy
from typing import Callable, Sequence, List, Union, Dict, Tuple, Optional
from functools import singledispatch

import numpy as np
import pandas as pd

VALID_KEYS = {
    "scale_x",
    "scale_y",
    "scale_z",
    "translate_x",
    "translate_y",
    "translate_z",
    "rotate_xy",
    "rotate_yz",
    "rotate_xz",
    "project",
}
OPERATIONS = {}
INDICES = {"x": 0, "y": 1, "z": 2, "xy": (0, 1), "yz": (1, 2), "xz": (0, 2)}

Conversion = Union[List[Tuple[str, Union[float, Dict]]], Callable]
XYCoords = Tuple[Union[float, Sequence[float]], Union[float, Sequence[float]]]
Convertable = Union[List, Tuple, pd.DataFrame, Sequence, np.array]


@singledispatch
def convert_coords(
    points: Convertable,
    conversion: Conversion,
    conversion_kwargs: Optional[dict] = None,
    **kwargs,
) -> Convertable:
    """
    Converts coordinates from one system to another

    Parameters
    ----------
    points : Array-like
        List of points to be converted. Can take a variety of formats, including tuples, lists, pandas DataFrames,
        and numpy arrays as long as they can be arranged into a numpy array with three columns for the X, Y, and Z
        coordinates.
    conversion : list-like or callable
        A set of instructions or callable for the coordinate conversion. If instructions are provided, they should
        be arrange in a two-dimensional list-like structure of key-value pairs (i.e., [["scale_x", 0.3048], ["scale_y",
        0.3048]]). See notes for acceptable keywords. If a callable is provided, it should take a three-column numpy
        array as input and output.
    conversion_kwargs : dict, optional
        Used if conversion is a callable. kwargs to be passed to the callable.

    Other Parameters
    ----------------
    x_in : str, optional
        Name of the input x-coordinate column (default="x")
    y_in : str, optional
        Name of the input y-coordinate column (default="y")
    z_in : str, optional
        Name of the input z-coordinate column (default="z")
    x_out : str, optional
        Name of the output x-coordinate column (default="x_conv")
    y_out : str, optional
        Name of the output y-coordinate column (default="y_conv")
    z_out : str, optional
        Name of the output z-coordinate column (default="z_conv")
    inplace : bool, optional
        Indicates whether the coordinate conversion should be done on the DataFrame in place or a new DataFrame should
        be returned. (Only when a DataFrame is provided as input.) (default=False)

    Notes
    -----
    The following describes the values to be provided with the keywords:\n
    scale_x : float
        Scale factor to apply to the x-coordinates\n
    scale_y : float
        Scale factor to apply to the x-coordinates\n
    scale_z : float
        Scale factor to apply to the x-coordinates\n
    translate_x : float
        Amount to move the coordinates in the x-direction\n
    translate_y : float
        Amount to move the coordinates in the y-direction\n
    translate_z : float
        Amount to move the coordinates in the z-direction\n
    rotate_xy : float
        Angle (in radians) to rotate the coordinates in the xy plane.
        Positive is CCW.\n
    rotate_xz : float
        Angle (in radians) to rotate the coordinates in the xz plane.
        Positive is CCW.\n
    rotate_yz : float
        Angle (in radians) to rotate the coordinates in the yz plane.
        Positive is CCW.\n
    project : dict
        Strings for the pyproj.Proj class projection, or a pyproj.Proj object (ex: "+init=EPSG:32611"). Keys in the
        dictionary should be "from" and "to"\n

    Returns
    -------
    points : Sequence
        Array-like object with the converted coordinates. If a DataFrame, list, or tuple were provided as input, then
        the same type will be returned. For all other formats, a numpy array will be returned.
    """
    if len(
        kwargs
    ):  # While the registered converters can support kwargs, the main one does not
        raise TypeError(f"Unexpected kwargs: {list(kwargs.keys())}")

    points = np.array(points, dtype=np.float64)
    if points.shape == (3,):
        points = points.reshape((1, 3))

    # Deal with a callable conversion
    if isinstance(conversion, Callable):
        try:
            if conversion_kwargs is not None:
                points = conversion(points, **conversion_kwargs)
            else:
                points = conversion(points)
        except Exception as e:
            raise RuntimeError(f"Conversion callable raised an exception: {e}")
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        if not (len(points.shape) == 2 and points.shape[1] == 3):
            raise TypeError(
                "Callable coordinate conversion must return a numpy array with 3 columns"
            )
        return points

    # Do some checking to make sure the conversion is kosher
    if not isinstance(conversion, Sequence):
        raise TypeError(f"conversion should be a list of transformations")
    try:
        if not conversion[0][0].lower() in VALID_KEYS:
            raise TypeError(f"conversion should be a list of transformations")
    except AttributeError:
        raise TypeError(f"conversion should be a list of transformations")

    # Go through each stage of the conversion
    for key, value in conversion:
        # Make sure the stage is valid
        key = key.lower()
        if key not in VALID_KEYS:
            raise ValueError(f"Invalid conversion operation: {key}")
        elif key == "project":
            points[:, 0], points[:, 1] = project(
                points[:, 0], points[:, 1], value["from"], value["to"]
            )
        else:
            try:
                value = float(value)
            except ValueError:
                raise TypeError(f"value must be a float for key: {key}, {value}")
            key = key.split("_")
            stage = OPERATIONS[key[0]]
            inds = INDICES[key[1]]
            stage(points, inds, value)

    if points.shape == (1, 3):
        points = points.reshape((3,))
    return points


@convert_coords.register(pd.DataFrame)
def _convert_df(
    df: pd.DataFrame,
    conversion,
    conversion_kwargs=None,
    x_in="x",
    y_in="y",
    z_in="z",
    x_out="x_conv",
    y_out="y_conv",
    z_out="z_conv",
    inplace=False,
):
    # Copy the input if not inplace
    if not inplace:
        df = copy(df)

    # Define the columns to do the conversion on
    points = np.array(df[[x_in, y_in, z_in]])
    points = convert_coords(points, conversion, conversion_kwargs)
    if points.shape == (3,):
        df[x_out] = points[0]
        df[y_out] = points[1]
        df[z_out] = points[2]
    else:
        df[x_out] = points[:, 0]
        df[y_out] = points[:, 1]
        df[z_out] = points[:, 2]
    return df


@convert_coords.register(tuple)
def _convert_tuple(points: tuple, conversion, conversion_kwargs=None):
    points = np.array(points)
    if len(points.shape) == 1:  # Single point
        if not points.shape[0] == 3:
            raise TypeError("points must be a tuple of three-dimensional points")
        return tuple(convert_coords(points, conversion, conversion_kwargs))
    elif points.shape[1] == 3:  # Multiple points
        converted = convert_coords(points, conversion, conversion_kwargs)
        points = []
        for p in converted:
            points.append(tuple(p))
        return tuple(points)
    else:
        raise TypeError("points must be a tuple of three-dimensional points")


@convert_coords.register(list)
def _convert_list(points: list, conversion, conversion_kwargs=None):
    points = np.array(points)
    if len(points.shape) == 1:  # Single point
        if not points.shape[0] == 3:
            raise TypeError("points must be a list of three-dimensional points")
        return list(convert_coords(points, conversion, conversion_kwargs))
    elif points.shape[1] == 3:  # Multiple points
        return list(convert_coords(points, conversion, conversion_kwargs))
    else:
        raise TypeError("points must be a list of three-dimensional points")


def rotate_points(
    x: Union[float, Sequence[float]], y: Union[float, Sequence[float]], ang: float
) -> XYCoords:  # around origin
    """
    Rotate a point about the origin
    Parameters
    ----------
    x : float or list of floats (required)
        X-coordinate(s)
    y : float or list of floats (required)
        Y-coordinate(s)
    ang : float (required)
        Rotation angle (radians)
    """
    x_temp = x * np.cos(ang) - y * np.sin(ang)
    y_temp = x * np.sin(ang) + y * np.cos(ang)
    return x_temp, y_temp


def project(x, y, fro, to):
    import pyproj

    if not isinstance(fro, pyproj.Proj):
        try:
            fro = pyproj.Proj(str(fro))
        except RuntimeError:
            raise ValueError(f"{fro} is not a valid projection")
    if not isinstance(to, pyproj.Proj):
        try:
            to = pyproj.Proj(str(to))
        except RuntimeError:
            raise ValueError(f"{to} is not a valid projection")
    return pyproj.transform(fro, to, x, y)


# Operations for coordinate conversions
def _scale(points, ind, value):
    points[:, ind] = points[:, ind] * value


OPERATIONS["scale"] = _scale


def _translate(points, ind, value):
    points[:, ind] = points[:, ind] + value


OPERATIONS["translate"] = _translate


def _rotate(points, inds, ang):
    points[:, inds[0]], points[:, inds[1]] = rotate_points(
        points[:, inds[0]], points[:, inds[1]], ang
    )


OPERATIONS["rotate"] = _rotate
