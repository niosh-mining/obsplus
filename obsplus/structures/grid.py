#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains an experimental class/functions for creating a
manipulating a grid for storing spatially variable data (for instance,
velocities, travel times, geologic properties, etc.). The interfaces for the
Grid class and its supporting functions are still under development and may
be subject to change without warning. Use at your own risk.
"""

from copy import copy
from typing import Sequence, Iterable, Optional, Tuple, List, Union
from numbers import Number, Integral
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata

from obsplus.conversions import convert_coords, Conversion
from obsplus.utils.misc import read_file


class Grid:
    """
    Class for generating/storing grids (compatible with NonLinLoc)

    This class is currently experimental and its interface may change.

    Parameters
    ----------
    base_name : str
        Path to the grid file (minus extension).
    gtype : str
        String describing the grid type. Common NonLinLoc formats are:
        "VELOCITY", "SLOWNESS", "VEL2", "SLOW2", "SLOW_LEN", "TIME", "TIME2D",
        "MISFIT", "ANGLE", "ANGLE2D"
    origin : list of float
        List of coordinates defining the origin using a right-handed coordinate
        system (X positive East, Y positive North, Z positive up).
        Functionality is only guaranteed for two- or three-dimensional grids
        at this time.
    spacing : list of float
        List of grid spacings in each direction. Functionality is only
        guaranteed for two- or three-dimensional grids at this time.
    num_cells : list of int, optional
        List of number of cells in each direction. Functionality is only
        guaranteed for two- or three-dimensional grids at this time. Must
        specify this or number of grid points.
    num_gps : list of int, optional
        List of number of grid points (number of cells + 1) in each direction.
        Functionality is only guaranteed for two-or three-dimensional grids at
        this time. Must specify this or number of cells.
    station : str, optional
        Name of station, if grid is specific to a seismic station (ex.
        travel-time grid) (default=None)

    Attributes
    ----------
    base_name : str
        Name for the grid for writing to file
    gtype : str
        Grid type descriptor
    header : dict
        Dictionary containing all of the parameters found in the grid header
    grid_points : list of numpy array
        Listing of grid coordinates along each dimension
    grid_map : numpy meshgrid
        Meshgrid mapping grid indices to physical space
    values : numpy array
        Array containing the values at each grid point
    """

    def __init__(
        self,
        base_name: str,
        gtype: str,
        origin: List[float],
        spacing: List[float],
        num_cells: Optional[List[int]] = None,
        num_gps: Optional[List[int]] = None,
        station: Optional[str] = None,
    ) -> None:
        # Conduct initial checks to make sure inputs are kosher
        origin, spacing, num_gps = self._check_grid_inputs(
            base_name=base_name,
            origin=origin,
            spacing=spacing,
            num_cells=num_cells,
            num_gps=num_gps,
        )
        self.base_name = base_name
        self.gtype = gtype
        self.header = {
            "num_gps": num_gps,
            "origin": origin,
            "spacing": spacing,
            "station": station,
            "gtype": gtype,
        }

        # Build the shape of the grid based on the provided geometry
        xmin = self.header["origin"]
        xmax = [
            xmin[i] + self.header["spacing"][i] * (self.header["num_gps"][i] - 1)
            for i in range(len(xmin))
        ]
        lims = zip(np.array(xmin), np.array(xmax), np.array(self.header["spacing"]))
        self.grid_points = [np.arange(x[0], x[1] + x[2] * 0.0001, x[2]) for x in lims]
        self.grid_map = np.meshgrid(*self.grid_points, indexing="ij")
        self.values = np.ones(self.header["num_gps"]) * -99
        return

    def get_value(self, point: Sequence[float], interpolate: bool = False) -> Number:
        """
        Method for retrieving a value at the closest cell to the specified
        point in the grid.

        Parameters
        ----------
        point : list of floats or ints
            Physical coordinates of point to seek out in the grid
        interpolate : bool, optional
            Flag to indicate whether to interpolate between grid cells.
            If False, returns value at the center of the nearest grid cell.
            If True, does a linear interpolation. Default is False.

        Returns
        -------
        value : Number
            The retrieved value
        """
        if interpolate:
            # Get the indices of the points immediately surrounding the point
            ind = _coord2surrounding_ind(point, self.grid_points)
            if len(ind.shape) == 1:
                # The point is outside of the grid... get the nearest point
                return self.get_value(point, interpolate=False)
            elif ind.shape[0] == 2:
                # Do a bilinear interpolation
                return _bilinear(point, ind, self.grid_points, self.values)
            elif ind.shape[0] == 3:
                # Do a trilinear interpolation
                return _trilinear(point, ind, self.grid_points, self.values)
            else:
                raise RuntimeError(
                    "Interpolation only supported for 2D or 3D grids at this time"
                )
        else:
            ind = self.get_index(point)
            value = self.values[ind]
            return value

    def get_index(self, point: Sequence[float]) -> tuple:
        """
        Method for retrieving index of the cell point belongs to.

        Parameters
        ----------
        point : list-like (required)
            Physical coordinates of point to seek out in the grid

        Returns
        -------
        tuple
            Index of the cell
        """
        return _coord2ind(point, self.grid_points)

    # --- Write grid to file
    def write(self, path: Optional[str] = None) -> None:
        """
        Method for writing the grid to a binary file.

        Parameters
        ----------
        path : str, optional
            Base path (minus extension) for the grid. Defaults to
            Grid.base_name if not provided.
        """
        # Determine the file names
        if path is None:
            path = self.base_name
        head_path = path + ".hdr"
        bin_path = path + ".buf"

        # Write the grid file
        write_header(
            head_path,
            self.header["num_gps"],
            self.header["origin"],
            self.header["spacing"],
            self.header["gtype"],
        )
        write_bin_grid(bin_path, self.values)
        return

    # --- Methods for plotting
    def plot_2d(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Method for plotting a 2D grid

        Other Parameters
        ----------------
        figsize : tuple
            Size of the figure. Default is (9, 9)
        cmap : str or matplotlib Colormap (or related)
            Colormap to use for displaying grid values. Default is "rainbow".
        alpha : float
            Transparency value for the colormap. Default is 0.5.
        legend_label : str
            Label to display on the colorbar. Default is "Value".
        shading : str
            Indicates whether to use shading when rendering. Acceptable values
            are "flat" for a solid color in each grid cell (default) or
            "gouraud" to apply Gouraud shading (see matplotlib docs).
        contour : bool
            Flag to indicate whether to plot contour lines instead of a colormap

        Returns
        -------
        fig : matplotlib Figure
            Figure containing the resultant plot axes
        ax : matplotlib Axes
            Axes containing the resultant plot
        """
        # Make sure the grid is kosher for this plotting method
        if not (len(self.values.shape) == 2):
            raise TypeError(
                "plot_2d method only works for 2D grid. Use plot_slice instead"
            )

        # Set necessary kwargs
        if "legend_label" not in kwargs:
            kwargs["legend_label"] = self.header["gtype"]

        # Generate the plot
        fig, ax = plt_grid(self.grid_map[0], self.grid_map[1], self.values, **kwargs)
        return fig, ax

    def plot_slice(
        self,
        layer: Number,
        layer_coord: str = "grid",
        orientation: int = 2,
        transpose: bool = False,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Method to plot a section of a 3D grid

        Parameters
        ----------
        layer : int or float
            Index or coordinate of the slice of the model to be plotted
        layer_coord : str, optional
            If set to "grid", then the value in layer should be the coordinate
            of the model slice in the grid space (default). If it is "ind",
            then it should be the index of the grid slice.
        orientation : int, optional
            Index of the dimension of to plot the cross-section in (e.x.,
            index=2 would plot an x-y view). Default is 2.
        transpose : bool (default=False)
            Indicates whether to flip the axes

        Other Parameters
        ----------------
        figsize : tuple
            Size of the figure. Default is (9, 9)
        cmap : str or matplotlib Colormap (or related)
            Colormap to use for displaying grid values. Default is "rainbow".
        alpha : float
            Transparency value for the colormap. Default is 0.5.
        legend_label : str
            Label to display on the colorbar. Default is "Value".
        shading : str
            Indicates whether to use shading when rendering. Acceptable values
            are "flat" for a solid color in each grid cell (default) or
            "gouraud" to apply Gouraud shading (see matplotlib docs).
        contour : bool
            Flag to indicate whether to plot contour lines instead of a colormap

        Returns
        -------
        fig : matplotlib Figure
            Figure containing the resultant plot axes
        ax : matplotlib Axes
            Axes containing the resultant plot

        Notes
        -----
        This will most likely only function as expected with a 3D grid.
        """
        # Make sure the grid is kosher for this plotting method
        if len(self.values.shape) < 3:
            raise TypeError(
                "plot_slice method only works for 3D grids. Use plot_2D instead"
            )

        if layer_coord == "grid":
            # More complicated than I thought it would be...
            grid_tuple = copy(self.header["origin"])
            grid_tuple[orientation] = layer
            layer = _coord2ind(grid_tuple, self.grid_points)[orientation]
        elif layer_coord == "ind":
            if not isinstance(layer, int):
                raise TypeError(
                    "Specified value is not a valid grid index (should be an integer)."
                )
        else:
            raise ValueError("Invalid option for layer_coord.")

        slice_tuple = [slice(None)] * len(self.grid_map)
        slice_tuple[orientation] = layer
        slice_tuple = tuple(slice_tuple)
        val_slice = self.values[slice_tuple]
        x = self.grid_map[orientation - 2][slice_tuple]
        y = self.grid_map[orientation - 1][slice_tuple]
        if (orientation - 2 < 0) and (orientation - 1 >= 0):
            x1 = y
            y = x
            x = x1
        if transpose:
            x1 = y
            y = x
            x = x1

        # Set necessary kwargs
        if "legend_label" not in kwargs:
            kwargs["legend_label"] = self.header["gtype"]

        # Generate the plot
        fig, ax = plt_grid(x, y, val_slice, **kwargs)
        return fig, ax

    # ----------------------- Internal Methods ----------------------- #
    def _check_grid_inputs(self, base_name, origin, spacing, num_cells, num_gps):
        """
        Internal method to validate Grid class inputs.

        Parameters
        ----------
        base_name : str
            Path to the grid file (minus extension)
        origin : list of floats
            List of coordinates defining the origin using a right-handed
            coordinate system (X positive East, Y positive North, Z positive
            up). Functionality is only guaranteed for three-dimensional grids
            at this time.
        spacing : list of floats
            List of grid spacing in each direction. Functionality is only
            guaranteed for three-dimensional grids at this time. Note that
            NonLinLoc velocity grids must have equal spacing in all directions.
        num_cells : list of ints
            List of number of cells in each direction. Functionality is only
            guaranteed for three-dimensional grids at this time. Must specify
            this or number of grid points.
        num_gps : list of ints
            List of number of grid points (number of cells + 1) in each
            direction. Functionality is only guaranteed for three-dimensional
            grids at this time. Must specify this or number of cells.

        Returns
        -------
        origin : list of floats
            Correctly-typed grid origin
        spacing : list of floats
            Correctly-typed grid spacing
        num_gps : list of ints
            Correctly-typed number of grid points along each dimension
        """
        # File path and grid type
        if base_name is None:
            raise ValueError(
                "You must specify a path for the grid file (minus the file "
                "extension)"
            )

        # Origin
        if origin is None:
            raise ValueError("You must specify an origin for the grid")
        self._check_iterable(origin, "origin")
        try:
            origin = [float(i) for i in origin]
        except ValueError:
            raise TypeError("origin coordinates must be floats")

        # Spacing
        if not spacing:
            raise ValueError("You must specify a spacing for the grid")
        self._check_iterable(spacing, "spacing")
        try:
            spacing = [float(i) for i in spacing]
        except ValueError:
            raise TypeError("grid spacings must be floats")

        # Number of cells/grid points
        if not num_cells and not num_gps:
            raise ValueError("num_cells or num_gps must be specified")
        if num_cells:
            self._check_iterable(num_cells, "num_cells")
            if not all([isinstance(i, Integral) for i in num_cells]):
                raise TypeError("num_cells values must be ints")
        if num_gps:
            self._check_iterable(num_gps, "num_gps")
            if not all([isinstance(i, Integral) for i in num_gps]):
                raise TypeError("num_gps values must be ints")
        if num_cells and num_gps:
            if not len(num_cells) == len(num_gps):
                raise ValueError("num_cells and num_gps must be the same shape")
            if not all(
                [val == (num_gps[ind] - 1) for (ind, val) in enumerate(num_cells)]
            ):
                raise ValueError(
                    "num_gps must be one greater than num_cells in each dimension"
                )
        if num_cells and not num_gps:
            num_gps = [i + 1 for i in num_cells]

        # Final checks
        if not len(origin) == len(spacing) == len(num_gps):
            raise ValueError(
                f"Grid dimensions must match. Provided dimensions: "
                f"Origin: {len(origin)} Spacing: {len(spacing)} "
                f"Number of Grid Points: {len(num_gps)}"
            )
        return origin, spacing, num_gps

    @staticmethod
    def _check_iterable(obj, name):
        if not (isinstance(obj, Iterable) and not isinstance(obj, str)):
            raise TypeError(f"{name} must be iterable")


# --------------- Internal functions for retrieving values from grids --------- #
def _coord2ind(point, xls):
    """
    Convert the spatial coordinates of a point to grid indeces\n

    Parameters
    ----------
    point : tuple
        Point to be converted to indices
    xls : numpy array
        Array describing the model geometry
    """
    # index of best fitting x point in L1 sense
    xind = [abs(point[num2] - xls[num2]).argmin() for num2 in range(len(xls))]
    return tuple(xind)


def _coord2surrounding_ind(point: Sequence[float], xls: np.array):
    """
    Get the indices of the points in a grid surrounding the specified point.

    If the point falls outside of the grid, fall back to returning the
    nearest point.

    Parameters
    ----------
    point : list of float
        Point for which to retrieve the surrounding indices
    xls : numpy array
        Array describing the grid geometry

    Returns
    -------
    tuple
        Min and max indices of the points surrounding the point in each
        dimension, or the nearest point if the point is not in the grid.
    """
    xind = np.array(
        [np.argpartition(abs(point[num] - x), 2)[0:2] for (num, x) in enumerate(xls)]
    )
    for num, x in enumerate(xls):
        if not min(x[xind[num]]) < point[num] < max(x[xind[num]]):
            return np.array(
                [abs(point[num] - x).argmin() for (num, x) in enumerate(xls)]
            )
        return xind


def _bilinear(point, inds, pts, vals):
    """
    Do a bilinear interpolation for the provided point
    """
    x1 = inds[0][0]
    x2 = inds[0][1]
    y1 = inds[1][0]
    y2 = inds[1][1]
    a = np.array(
        [
            [1, pts[0][x1], pts[1][y1], pts[0][x1] * pts[1][y1]],
            [1, pts[0][x1], pts[1][y2], pts[0][x1] * pts[1][y2]],
            [1, pts[0][x2], pts[1][y1], pts[0][x2] * pts[1][y1]],
            [1, pts[0][x2], pts[1][y2], pts[0][x2] * pts[1][y2]],
        ]
    )
    d = np.array([[vals[x1, y1]], [vals[x1, y2]], [vals[x2, y1]], [vals[x2, y2]]])
    m = np.linalg.inv(a).dot(d)
    out = m[0] + m[1] * point[0] + m[2] * point[1] + m[3] * point[0] * point[1]
    return out[0]


def _trilinear(point, inds, pts, vals):
    """
    Do a trilinear interpolation for the provided point
    """
    xs = pts[0]
    ys = pts[1]
    zs = pts[2]
    x1 = inds[0][0]
    x2 = inds[0][1]
    y1 = inds[1][0]
    y2 = inds[1][1]
    z1 = inds[2][0]
    z2 = inds[2][1]
    a = np.array(
        [
            [
                1,
                xs[x1],
                ys[y1],
                zs[z1],
                xs[x1] * ys[y1],
                xs[x1] * zs[z1],
                ys[y1] * zs[z1],
                xs[x1] * ys[y1] * zs[z1],
            ],
            [
                1,
                xs[x2],
                ys[y1],
                zs[z1],
                xs[x2] * ys[y1],
                xs[x2] * zs[z1],
                ys[y1] * zs[z1],
                xs[x2] * ys[y1] * zs[z1],
            ],
            [
                1,
                xs[x1],
                ys[y2],
                zs[z1],
                xs[x1] * ys[y2],
                xs[x1] * zs[z1],
                ys[y2] * zs[z1],
                xs[x1] * ys[y2] * zs[z1],
            ],
            [
                1,
                xs[x2],
                ys[y2],
                zs[z1],
                xs[x2] * ys[y2],
                xs[x2] * zs[z1],
                ys[y2] * zs[z1],
                xs[x2] * ys[y2] * zs[z1],
            ],
            [
                1,
                xs[x1],
                ys[y1],
                zs[z2],
                xs[x1] * ys[y1],
                xs[x1] * zs[z2],
                ys[y1] * zs[z2],
                xs[x1] * ys[y1] * zs[z2],
            ],
            [
                1,
                xs[x2],
                ys[y1],
                zs[z2],
                xs[x2] * ys[y1],
                xs[x2] * zs[z2],
                ys[y1] * zs[z2],
                xs[x2] * ys[y1] * zs[z2],
            ],
            [
                1,
                xs[x1],
                ys[y2],
                zs[z2],
                xs[x1] * ys[y2],
                xs[x1] * zs[z2],
                ys[y2] * zs[z2],
                xs[x1] * ys[y2] * zs[z2],
            ],
            [
                1,
                xs[x2],
                ys[y2],
                zs[z2],
                xs[x2] * ys[y2],
                xs[x2] * zs[z2],
                ys[y2] * zs[z2],
                xs[x2] * ys[y2] * zs[z2],
            ],
        ]
    )
    d = np.array(
        [
            [vals[x1, y1, z1]],
            [vals[x2, y1, z1]],
            [vals[x1, y2, z1]],
            [vals[x2, y2, z1]],
            [vals[x1, y1, z2]],
            [vals[x2, y1, z2]],
            [vals[x1, y2, z2]],
            [vals[x2, y2, z2]],
        ]
    )
    m = np.linalg.inv(a).dot(d)
    return (
        m[0]
        + m[1] * point[0]
        + m[2] * point[1]
        + m[3] * point[2]
        + m[4] * point[0] * point[1]
        + m[5] * point[0] * point[2]
        + m[6] * point[1] * point[2]
        + m[7] * point[0] * point[1] * point[2]
    )[0]


# --------------- Functions for building/reading grids --------------- #
def load_grid(base_name: str, gtype: Optional[str] = None) -> Grid:
    """
    Function for reading a binary binary grid

    Parameters
    ----------
    base_name : str
        Name of the grid file (minus extension)
    gtype : str, optional
        If specified, verify the grid is the expected format

    Returns
    -------
    grid : Grid
        Grid object containing the binary grid
    """
    # Make sure the grid files exist
    header_name = base_name + ".hdr"
    binary_name = base_name + ".buf"
    if not os.path.isfile(header_name):
        raise IOError(f"Binary grid header file {header_name} does not exist")
    if not os.path.isfile(binary_name):
        raise IOError(f"Binary grid buffer file {binary_name} does not exist")

    # Read the header and use it to size the grid
    header = read_header(header_name, gtype)
    # Initialize a grid Grid object
    grid = Grid(
        base_name,
        gtype=header["gtype"],
        origin=header["origin"],
        spacing=header["spacing"],
        num_gps=header["num_gps"],
        station=header["station"],
    )

    # Populate the grid with values from the binary file
    grid = read_bin_grid(binary_name, grid)
    return grid


def read_header(path: str, gtype: Optional[str] = None) -> dict:
    """
    Function for reading NonLinLoc binary grid header files

    Parameters
    ----------
    path : str
        Path to the header file
    gtype : str, optional
        If specified, verify the grid is the expected format

    Returns
    -------
    header_data : dict
        Contents of the header file
    """
    with open(path, "r") as headfile:
        header = headfile.read().split()
    if not header[9] == gtype:
        msg = f"Grid format does not match specified grid type: {gtype}"
        raise TypeError(msg)
    num_gps = [int(header[0]), int(header[1]), int(header[2])]
    origin = [float(header[3]), float(header[4]), float(header[5])]
    spacing = [float(header[6]), float(header[7]), float(header[8])]
    if header[9] in ["TIME", "TIME2D", "ANGLE", "ANGLE2D"]:
        station = {
            "name": header[11],
            "x": float(header[12]),
            "y": float(header[13]),
            "z": -1 * float(header[14]),
        }
    else:
        station = None
    gtype = header[9]
    # Fix origin to correspond to a right-handed system with positive z up
    origin[2] = -1 * origin[2] - spacing[2] * (num_gps[2] - 1)
    header_data = {
        "num_gps": num_gps,
        "origin": origin,
        "spacing": spacing,
        "station": station,
        "gtype": gtype,
    }
    return header_data


def read_bin_grid(path: str, grid: Grid) -> Grid:
    """
    Function for reading in a NonLinLoc binary grid

    Parameters
    ----------
    path : str
        Path to the binary buffer
    grid : Grid
        Grid to dump the values into

    Returns
    -------
    grid : Grid
        Grid as read from the file.
    """
    values = np.fromfile(path, dtype=np.float32)
    values = np.flip(values.reshape(grid.header["num_gps"]), axis=2)
    grid.values = values
    return grid


def write_header(
    path: str, gps: Sequence[int], origin: Sequence[float], spacing: float, gtype: str
) -> None:
    """
    Function for writing a NonLinLoc binary grid header

    Parameters
    ----------
    path : str
        Path for the header file
    gps : list of int
        Number of grid points in the X, Y, and Z directions
    origin : list of float
        X, Y, and Z coordinates of the grid origin (Z is positive up).
        Coordinates must be in the NonLinLoc coordinate system and must be in
        kilometers.
    spacing : float
        Size of the grid cells in kilometers. Grid cells must be square to be
        compatible with NonLinLoc.
    gtype : str
        Type of grid being written. Common options are: "VELOCITY", "SLOWNESS",
        "VEL2", "SLOW2", "SLOW_LEN", "TIME", "TIME2D", "MISFIT", "ANGLE", "ANGLE2D"

    Notes
    -----
    It is important to note that while the obsplus Grid objects here created
    such that Z is positive up, in order to be compatible with NonLinLoc, the
    origin must be swapped when it is written so Z is positive down.
    """
    # Swap origin to be a left-handed system with positive z down
    origin_z = -1 * (origin[2] + spacing[2] * (gps[2] - 1))
    header_gps = f"{gps[0]} {gps[1]} {gps[2]} "
    header_origin = f"{origin[0]:0.3f} {origin[1]:0.3f} {origin_z:0.3f} "
    header_spacing = f"{spacing[0]:0.3f} {spacing[1]:0.3f} {spacing[2]:0.3f} "
    header_remainder = f"{gtype} FLOAT\nTRANSFORM NONE\n"
    header = "".join([header_gps, header_origin, header_spacing, header_remainder])
    # Check to make sure directory exists, if not, create it...
    direc = os.path.dirname(path)
    if os.path.isfile(direc):
        raise IOError("{direc} is not a directory")
    elif not os.path.isdir(direc):
        os.makedirs(direc)
    with open(path, "w") as f:
        f.write(header)
    return


def write_bin_grid(path: str, grid: np.array) -> None:
    """
    Function for writing a NonLinLoc binary grid

    Parameters
    ----------
    path : str
        Path for the binary file
    grid : numpy array
        Three-dimensional numpy array containing the values for the grid.
    """
    grid = np.flip(grid, axis=2).reshape((1, grid.size))
    grid.astype("float32").tofile(path)


# ---------- Utilities for manipulating the values of a grid --------- #
def apply_layers(grid: Grid, layers: List[List[float]]) -> None:
    """
    Function for applying velocities from a 1D layered model.

    Parameters
    ----------
    grid : Grid
        Grid on which to apply the layers
    layers : list-like
        List of layers of the form [(value1, elevation1), (value2,
        elevation2)] to  apply to the grid
    """
    # For each layer in vel_input, apply the specified velocity
    for elev in layers:
        if not (isinstance(elev, Sequence) and not isinstance(elev, str)):
            raise TypeError(
                "velocity input should be a 2D list of values and elevations"
            )
        grid.values[:, :, grid.grid_points[2] <= elev[1]] = elev[0]


def apply_rectangles(
    grid: Grid,
    rectangles: Union[pd.DataFrame, str],
    tol: float = 1e-6,
    conversion: Optional[Conversion] = None,
    conversion_kwargs: Optional[dict] = None,
) -> None:
    """
    Function for perturbing grid values in rectangular regions

    Parameters
    ----------
    grid : Grid
        Grid on which to apply the layers
    rectangles : pandas DataFrame
        List of velocity perturbations to apply. Required columns
        include ["DELTA", "XMIN", "XMAX", "YMIN", "YMAX"] where "DELTA" is
        some percentage of the current grid value
    tol : float
        Value to add to rectangle coordinates to prevent rounding
        errors. This value should be small relative to the grid
        spacing.
    conversion : list-like or callable, optional
        See obsplus.conversions.convert_coords
    conversion_kwargs : dict, optional
        See obsplus.conversions.convert_coords
    """
    if isinstance(rectangles, str):
        path = rectangles
        if not os.path.isfile(path):
            raise OSError(f"rectangles file does not exist: {path}")
        rectangles = read_file(path)
        cols = {"delta", "xmin", "ymin", "zmin", "xmax", "ymax", "zmax"}
        if not cols.issubset(rectangles.columns):
            raise IOError(f"{path} is not a valid rectangles file")
    elif isinstance(rectangles, pd.DataFrame):
        pass
    else:
        raise TypeError("rectangles must be a pandas DataFrame")
    if conversion:
        for num, r in rectangles.iterrows():
            rectangles.loc[num, ["xmin", "ymin", "zmin"]] = convert_coords(
                r[["xmin", "ymin", "zmin"]],
                conversion=conversion,
                conversion_kwargs=conversion_kwargs,
            )
            rectangles.loc[num, ["xmax", "ymax", "zmax"]] = convert_coords(
                r[["xmax", "ymax", "zmax"]],
                conversion=conversion,
                conversion_kwargs=conversion_kwargs,
            )
    v = grid.values
    gmap = grid.grid_map
    for num, zone in rectangles.iterrows():
        # apply the perturbation
        delta = 0.01 * zone.delta
        xmask = (gmap[0] >= zone.xmin - tol) & (gmap[0] <= zone.xmax + tol)
        ymask = (gmap[1] >= zone.ymin - tol) & (gmap[1] <= zone.ymax + tol)
        zmask = (gmap[2] >= zone.zmin - tol) & (gmap[2] <= zone.zmax + tol)
        v[xmask & ymask & zmask] = v[xmask & ymask & zmask] * delta
    return


def apply_topo(
    grid: Grid,
    topo_points: Union[str, pd.DataFrame],
    air: float = 0.343,
    method: str = "nearest",
    conversion: Optional[Conversion] = None,
    conversion_kwargs: Optional[dict] = None,
    tolerance: float = 1e-6,
    buffer: int = 0,
    topo_label: str = "TOPO",
) -> Grid:
    """
    Function for applying a topographic surface to the model

    Parameters
    ----------
    grid : Grid
        Grid on which to apply the layers
    topo_points : pandas DataFrame or path to csv or dxf file
        Input containing the topography data
    air : float, optional
        Value to assign to the "air" blocks (blocks above the topography).
        Default is 0.343 km/s.
    method : str, optional
        Method used by scipy's griddata to interpolate the topography grid.
        Acceptable values include: "nearest" (default), "linear", and "cubic"
    conversion : list-like or callable, optional
        See obsplus.conversions.convert_coords
    conversion_kwargs : dict, optional
        See obsplus.conversions.convert_coords
    tolerance : float, optional
        Should be small relative to the grid size. Deals with those pesky
        rounding errors. (Default=1e-6)
    buffer : int, optional
        Number of cells above the topography to extend the "rock" values
        (i.e., the values that are below the topography) (default=0)
    topo_label : str, optional
        Label to assign the 2D grid that is created from topo_points
        (default="TOPO")

    Returns
    -------
    topo : Grid
        2D grid created from topo_points

    Notes
    -----
    If the input is a CSV file or pandas DataFrame, the following columns are
    required: ["x", "y", "z"]. If the input is a dxf file, the dxf should not
    contain any data other than the topography (in the form of LWPOLYLINEs,
    LINEs, POLYLINES, POINTS, and/or 3DFACEs) and must have the elevation data
    stored in the entities (Z coordinates shouldn't be 0.0).\n
    It should be noted that bizarre results may occur if the topo data does
    not extend beyond the grid region.\n
    """
    # Do some basic error checking and read in the topo data
    if isinstance(topo_points, pd.DataFrame):
        topo = topo_points
        if not {"x", "y", "z"}.issubset(topo.columns):
            raise KeyError(
                "topo_points must contain the following columns: ['x', 'y', 'z']"
            )
    elif isinstance(topo_points, str):
        if not os.path.isfile(topo_points):
            raise OSError(f"topo file does not exist: {topo_points}")
        topo = read_file(topo_points, funcs=(pd.read_csv, _read_topo_dxf))
        if not {"x", "y", "z"}.issubset(topo.columns):
            raise IOError(f"{topo_points} is not a valid topo file")
    else:
        raise TypeError("An invalid topo_points was provided to apply_topo")
    if not {"x", "y", "z"}.issubset(topo.columns):
        raise KeyError(
            "topo_points must contain the following columns: ['x', 'y', 'z']"
        )

    if method not in ["nearest", "linear", "cubic"]:
        raise ValueError(f"Unsupported interpolation format: {method}")

    # Apply a coordinate conversion, if necessary
    if conversion:
        topo = convert_coords(
            topo,
            conversion,
            x_in="x",
            y_in="y",
            z_in="z",
            x_out="x",
            y_out="y",
            z_out="z",
            conversion_kwargs=conversion_kwargs,
        )

    # Create a 2D grid space for the topo map
    grid_x, grid_y = np.mgrid[
        grid.grid_points[0][0] : grid.grid_points[0][-1] : grid.header["num_gps"][0]
        * 1j,
        grid.grid_points[1][0] : grid.grid_points[1][-1] : grid.header["num_gps"][1]
        * 1j,
    ]
    # Interpolate over the grid
    nearest_grid = griddata(
        np.array(topo[["x", "y"]]),
        np.array(topo["z"]),
        (grid_x, grid_y),
        method="nearest",
    )
    if method == "nearest":
        topo_grid = nearest_grid
    else:
        topo_grid = griddata(
            np.array(topo[["x", "y"]]),
            np.array(topo["z"]),
            (grid_x, grid_y),
            method=method,
        )
        mask = np.isnan(topo_grid)
        topo_grid[mask] = nearest_grid[mask]
    topo_grid = topo_grid - tolerance
    topo_grid = np.ndarray.astype(topo_grid, np.float64)
    topo = Grid(
        base_name=grid.base_name + "_topo",
        gtype=topo_label,
        origin=grid.header["origin"][0:2],
        num_gps=grid.header["num_gps"][0:2],
        spacing=grid.header["spacing"][0:2],
    )
    topo.values = topo_grid
    # Overlay the topo grid on the velocity grid
    elevs = grid.grid_map[:][:][2]
    # Optionally add extra cells above the topo layer to deal with the ways
    # some programs interpolate between cells
    mask = np.array(
        [
            (elevs[:, :, i] > topo_grid + buffer * grid.header["spacing"][2])
            for i in range(elevs.shape[2])
        ]
    )
    mask = np.swapaxes(mask, 0, 1)
    mask = np.swapaxes(mask, 1, 2)
    grid.values[mask] = air
    return topo


# --------------------- Grid plotting utilities ---------------------- #
def plt_grid(
    x1_coords: List[float],
    x2_coords: List[float],
    values: List[float],
    contour: bool = False,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Function to plot a colormap of a grid

    Parameters
    ----------
    x1_coords : array-like
        Coordinates of grid points along the horizontal axis of the plot
    x2_coords : array-like
        Coordinates of grid points along the vertical axis of the plot
    values : array-like
        Values stored in the grid
    contour : bool, optional
        Flag indicating whether to plot contour lines instead of a colormap
        (default=False).

    Other Parameters
    ----------------
    figsize : tuple
        Size of the figure (default=(9,9))
    cmap : str or matplotlib Colormap (or related)
        Colormap to use for displaying grid values (default="rainbow")
    alpha : float
        Transparency value for the colormap (default=0.5)
    legend_label : str
        Label to display on the colorbar (default="Value")
    shading : str
        Indicates whether to use shading when rendering. Acceptable values are
        "flat" for a solid color in each grid cell (default) or "gouraud" to
        apply Gouraud shading (see matplotlib docs)

    Returns
    -------
    fig : matplotlib Figure
        Figure containing the resultant plot axes
    ax : matplotlib Axes
        Axes containing the resultant plot
    """
    figsize = kwargs.pop("figsize", (9, 9))
    cmap = kwargs.pop("cmap", "rainbow")
    alpha = kwargs.pop("alpha", 0.5)
    legend_label = kwargs.pop("legend_label", "Value")
    shading = kwargs.pop("shading", "flat")

    fig = plt.figure(figsize=figsize)

    # Define the plot grid and create a new axes

    gs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=[10, 0.5],
        height_ratios=[10, 0.5],
        wspace=0.05,
        hspace=0.05,
    )
    ax = fig.add_subplot(gs[0])

    # Plot the grid with the appropriate user args
    if not contour:
        cmesh = ax.pcolormesh(
            x1_coords,
            x2_coords,
            values,
            cmap=cmap,
            alpha=alpha,
            shading=shading,
            **kwargs,
        )
    else:
        cmesh = ax.contour(x1_coords, x2_coords, values, colors=cmap, **kwargs)
        ax.clabel(cmesh, colors="k", fmt="%2.2f", fontsize=12)
    # Make the plot look nice and add a legend
    ax.xaxis.tick_top()

    ax2 = fig.add_subplot(gs[2])
    fig.colorbar(cmesh, cax=ax2, orientation="horizontal", label=legend_label)

    # Make the plot look nice
    ax.set_aspect("equal")
    return fig, ax


def grid_cross(
    grid: Grid, coord: float, direction: str = "X"
) -> Tuple[np.array, np.array]:
    """
    Function for returning a profile from a two-dimensional grid

    Parameters
    ----------
    grid : Grid
        Grid from which to pull the values
    coord : float
        Coordinate along which to get the profile
    direction : str, optional
        Direction along which to get the profile. Possible values are "X"
        for a profile that parallel to the x-axis (default) and "Y" for a profile
        that is parallel to the y-axis.

    Returns
    -------
    points : np.array
        List of coordinates along the profile
    values : np.array
        Values at each coordinate
    """
    if not len(grid.grid_points) == 2:
        raise TypeError("grid must be a 2D Grid")
    points = {"X": grid.grid_points[0], "Y": grid.grid_points[1]}

    if (coord < points[direction][0]) and (coord > points[direction][-1]):
        raise ValueError(
            f"coord is outside grid bounds: {coord} not in "
            f"({points[direction][0]}, {points[direction][-1]})"
        )
    elif direction == "X":
        # Slice the grid in the X-direction at that the provided Y-coordinate
        dummy = points["X"][0]
        ind = grid.get_index((dummy, coord))[1]
        values = grid.values[:, ind]
        return points[direction], values
    elif direction == "Y":
        # Slice the grid in the Y-direction at the provided X-coordinate
        dummy = points["Y"][0]
        ind = grid.get_index((coord, dummy))[0]
        values = grid.values[ind, :]
        return points[direction], values
    else:
        raise ValueError(f"Unknown direction: {direction}")


# ------------- Internal functions for handling dxfs ----------------- #
def _read_topo_dxf(dxf, line_end="\n"):
    """ Function for parsing topography information from a dxf file """
    with open(dxf, "r") as f:
        parsed = f.read()
    # Split by section
    parsed = parsed.split(f"  0{line_end}SECTION{line_end}")
    # Get the entities and split them up
    try:
        parsed = parsed[5].split(
            f"{line_end}  0{line_end}"
        )  # Technically dangerous... should seek out the entities section
    except IndexError:
        raise IOError(f"Invalid dxf file: {dxf}")
    parsed.pop(0)
    parsed.pop(-1)
    entities = []
    enum = enumerate(parsed)
    for num, entity in enum:
        records = entity.split(line_end)
        if records[0] == "LWPOLYLINE":
            records = _entity_boilerplate(records)
            # Get the elevation of the polyline
            try:
                elev = np.float64(records.loc[records.CODE == " 38"].iloc[0].VALUE)
            except IndexError:
                continue
            # Get a table of the points
            inlist = [" 10", " 20"]
            entity = _reshape_points(records, inlist, use_z=False)
            # Append the elevation
            entity["z"] = elev
            # Append the parsed polyline to the list of parsed entities
            entities.append(entity)
        elif records[0] == "LINE":
            records = _entity_boilerplate(records)
            # Append table of endpoints to list of parsed entities
            inlist = [" 10", " 20", " 30", " 11", " 21", " 31"]
            entities.append(_reshape_points(records, inlist))
        elif records[0] == "POLYLINE":
            # Loop through the next entities until reach something
            # other than a vertex
            points = []
            for item in parsed[(num + 1) :]:
                records = item.split(line_end)
                if records[0] != "VERTEX":
                    # If the entity is not a vertex, then reach end of
                    # polyline
                    break
                else:
                    # Otherwise, move the enumerator forward and add
                    # the parsed vertex to the point list
                    _ = next(enum)
                    points.append(_parse_pointlike(records))
            # Take the points and merge them into a single df and
            # append to the list of parsed entities
            entities.append(pd.concat(points))
        elif records[0] == "POINT":
            # Append the parsed point coord to the list of parsed entities
            entities.append(_parse_pointlike(records))
        elif records[0] == "TEXT":
            # Append the parsed text coord to the list of parsed entities
            entities.append(_parse_pointlike(records))
        elif records[0] == "3DFACE":
            records = _entity_boilerplate(records)
            # Append table of endpoints to list of parsed entities
            inlist = [
                " 10",
                " 20",
                " 30",
                " 11",
                " 21",
                " 31",
                " 12",
                " 22",
                " 32",
                " 13",
                " 23",
                " 33",
            ]
            entities.append(_reshape_points(records, inlist))
        else:
            pass
    entities = pd.concat(entities)
    return entities


def _entity_boilerplate(entity):
    """ Internal function for parsing an entity from a dxf file """
    # Parse into a key, value DataFrame
    e = np.array(entity[1:])
    if not (len(e) % 2) == 0:
        raise IOError(f"Corrupt dxf file while parsing {entity[0]}")
    e = e.reshape((len(e) // 2, 2))
    return pd.DataFrame(e, columns=["CODE", "VALUE"], dtype="object")


def _parse_pointlike(entity):
    """ Internal function for parsing a point-like entity from a dxf file """
    entity = _entity_boilerplate(entity)
    # Retrieve the X, Y, and Z coordinates of the point
    inlist = [" 10", " 20", " 30"]
    return _reshape_points(entity, inlist)


def _reshape_points(df, inlist, use_z=True):
    """ Internal function for properly reshaping a point entity """
    # Drop all of the extraneous stuff
    points = df.loc[df.CODE.isin(inlist)]
    # Reshape the series as a table of X, Y, and Z (optional) coordinates
    points = np.array(points.VALUE, dtype=np.float64)
    if use_z:
        dim = 3
        cols = ["x", "y", "z"]
    else:
        dim = 2
        cols = ["x", "y"]
    if not (len(points) % dim) == 0:
        handle = df.loc[df.CODE == "  5"].iloc[0].VALUE
        raise IOError(f"Corrupt dxf file crashed while parsing entity: {handle}")
    return pd.DataFrame(points.reshape((len(points) // dim, dim)), columns=cols)
