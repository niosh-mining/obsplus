#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 10:13:03 2017

@author: shawn
"""

from copy import copy
from typing import Iterable, Callable
import os.path
import warnings
from numbers import Integral

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata

from obsplus.utils import read_file

try:
    import pyproj
except ModuleNotFoundError:
    warnings.warn("pyproj is not installed on this system")


class Grid(object):
    """
    Class for generating/storing grids (compatible with NonLinLoc)

    Parameters
    ----------
    base_name : str (required)
        Name for the velocity model for writing to file. Path to the grid
        file (minus extension).
    gtype : str (required)
        String describing the grid type. Common NonLinLoc formats are:
        "VELOCITY", "SLOWNESS", "VEL2", "SLOW2", "SLOW_LEN", "TIME",
        "TIME2D", "PROB_DENSITY", "MISFIT", "ANGLE", "ANGLE2D"
    origin : list-like (required)
        List of coordinates defining the origin using a right-handed
        coordinate system (X positive East, Y positive North, Z positive
        up). Functionality is only guaranteed for three-dimensional grids
        at this time.
    spacing : list-like (required)
        List of grid spacings in each direction. Functionality is only
        guaranteed for three-dimensional grids at this time. Note that
        NonLinLoc velocity grids must have equal spacing in all
        directions.
    num_cells : list-like (int) (required)
        List of number of cells in each direction. Functionality is only
        guaranteed for three-dimensional grids at this time. Must specify
        this or number of grid points.
    num_gps : list-like (int) (required)
        List of number of grid points (number of cells + 1) in each
        direction. Functionality is only guaranteed for three-dimensional
        grids at this time. Must specify this or number of cells.
    station : list-like (default=None)
        Station the grid is specific to (required for certain grid types,
        such as travel-time grids)

    Attributes
    ----------
    base_name : str
        Name for the grid for writing to file
    gtype : str
        Grid type descriptor
    header : dict
        Dictionary containing all of the parameters found in the grid
        header
    grid_points : list of numpy array
        Listing of grid coordinates along each dimension
    grid_map : numpy meshgrid
        Meshgrid mapping grid indices to physical space
    values : numpy array
        Array containing the values at each grid point

    Notes
    -----
    If creating a "VELOCITY" or "VEL2" grid, the VelocityGrid object
    should be used.
    """

    def __init__(
        self,
        base_name=None,
        gtype=None,
        origin=None,
        spacing=None,
        num_cells=None,
        num_gps=None,
        station=None,
    ):
        # Conduct initial checks to make sure inputs are kosher
        origin, spacing, num_gps = self._check_grid_inputs(
            base_name=base_name,
            gtype=gtype,
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

    def get_value(self, point, interpolate=True):
        """
        Method for retrieving a value at the closest cell to the specified
        point in the grid (eventually want to update this to interpolate?)

        point : list-like (required)
            Physical coordinates of point to seek out in the grid
        interpolate : bool (default=True)
            Flag to indicate whether to interpolate between grid cells. If
            False, returns value at the center of the nearest grid cell.
            (Not yet implemented, not sure how or whether it should be
            implemented...)
        """
        if interpolate:
            raise Exception("Logic not yet developed to interpolate grid values")

        ind = self.get_index(point)
        value = self.values[ind]
        return value

    def get_index(self, point):
        """
        Method for retrieving index of the cell point belongs to.

        point : list-like (required)
            Physical coordinates of point to seek out in the grid
        """
        return coord2ind(point, self.grid_points)

    # --- Write grid to file
    def write(self, path=None):
        """
        Method for writing the grid to a binary file

        Parameters
        ----------
        path : str (default=Grid.base_name)
            Base path (minus extension) for the grid
        """
        # Determine the file names
        if path is None:
            path = self.base_name
        head_path = path + ".hdr"
        if self.header["gtype"] == "PROB_DENSITY":
            bin_path = path + ".scat"
        else:
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
    def plot_2d(self, **kwargs):
        """
        Method for plotting a 2D grid

        Kwargs
        ------
        figsize : tuple (default=(9,9))
            Size of the figure
        cmap : str or matplotlib Colormap (or related) (default="rainbow")
            Colormap to use for displaying grid values
        alpha : float (default=0.5)
            Transparency value for the colormap
        legend_label : str (default="Value")
            Label to display on the colorbar
        shading : str (default="flat")
            Indicates whether to use shading when rendering. Acceptable
            values are "flat" for a solid color in each grid cell or
            "gouraud" to apply Gouraud shading (see matplotlib docs)

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

        fig.show()
        return fig, ax

    def plot_slice(self, layer, layer_coord="grid", orien=2, transpose=False, **kwargs):
        """
        Method to plot a section of a 3D grid

        Parameters
        ----------
        layer : int or float
            Index or coordinate of the slice of the model to be plotted
        layer_coord : str (default="grid")
            If set to "grid", then the value in layer should be the
            coordinate of the model slice in the grid space. If it is
            "ind", then it should be the index of the grid slice.
            (Want to add functionality to allow for coordinate
            conversion between real-space and grid space?)
        orien : int (default=2)
            Index of the dimension of to plot the cross-section in (e.x.,
            index=2 would plot an x-y view)
        transpose : bool (default=False)
            Indicates whether to flip the axes

        Kwargs
        ------
        figsize : tuple (default=(9,9))
            Size of the figure
        cmap : str or matplotlib Colormap (or related) (default="rainbow")
            Colormap to use for displaying grid values
        alpha : float (default=0.5)
            Transparency value for the colormap
        legend_label : str (default="Value")
            Label to display on the colorbar
        shading : str (default="flat")
            Indicates whether to use shading when rendering. Acceptable
            values are "flat" for a solid color in each grid cell or
            "gouraud" to apply Gouraud shading (see matplotlib docs)

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
            grid_tuple[orien] = layer
            layer = coord2ind(grid_tuple, self.grid_points)[orien]
        elif layer_coord == "ind":
            if not isinstance(layer, int):
                raise TypeError(
                    "Specified value is not a valid grid index "
                    "(should be an integer)."
                )
        else:
            raise ValueError("Invalid option for layer_coord.")

        veltuple = [slice(None)] * len(self.grid_map)
        veltuple[orien] = layer
        veltuple = tuple(veltuple)
        velslice = self.values[veltuple]
        x = self.grid_map[orien - 2][veltuple]
        y = self.grid_map[orien - 1][veltuple]
        if (orien - 2 < 0) and (orien - 1 >= 0):
            x1 = y
            y = x
            x = x1
        if transpose:
            # velslice = velslice.transpose()
            x1 = y
            y = x
            x = x1

        # Set necessary kwargs
        if "legend_label" not in kwargs:
            kwargs["legend_label"] = self.header["gtype"]

        # Generate the plot
        fig, ax = plt_grid(x, y, velslice, **kwargs)
        if kwargs.get("show", True):
            fig.show()
        return fig, ax

    # ----------------------- Internal Methods ----------------------- #
    def _check_grid_inputs(self, base_name, gtype, origin, spacing, num_cells, num_gps):
        """
        Internal method to validate Grid class inputs\n

        Parameters
        ----------
        base_name : str (required)
            Path to the grid file (minus extension)
        gtype : str (required)
            String describing the grid type.
        origin : list-like (required)
            List of coordinates defining the origin using a right-handed
            coordinate system (X positive East, Y positive North, Z positive
            up). Functionality is only guaranteed for three-dimensional grids
            at this time.
        spacing : list-like (required)
            List of grid spacing in each direction. Functionality is only
            guaranteed for three-dimensional grids at this time. Note that
            NonLinLoc velocity grids must have equal spacing in all
            directions.
        num_cells : list-like (int) (required)
            List of number of cells in each direction. Functionality is only
            guaranteed for three-dimensional grids at this time. Must specify
            this or number of grid points.
        num_gps : list-like (int) (required)
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
                "You must specify a path for the grid file (minus"
                " the file extension)"
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
            for ind, val in enumerate(num_cells):
                if not all(
                    [val == (num_gps[ind] - 1) for (ind, val) in enumerate(num_cells)]
                ):
                    raise ValueError(
                        "num_gps must be one greater than "
                        "num_cells in each dimension"
                    )
        if num_cells and not num_gps:
            num_gps = [i + 1 for i in num_cells]

        # Final checks
        if not len(origin) == len(spacing) == len(num_gps):
            raise ValueError(
                "Grid dimensions must match. Provided dimensions:"
                f" Origin: {len(origin)} Spacing: {len(spacing)} "
                f"Number of Grid Points: {len(num_gps)}"
            )
        return origin, spacing, num_gps

    @staticmethod
    def _check_iterable(obj, name):
        if not (isinstance(obj, Iterable) and not isinstance(obj, str)):
            raise TypeError(f"{name} must be iterable")


# --------------- Functions for retrieving values from grids --------- #
def coord2ind(point, xls):
    """
    Convert the spatial coordinates of a point to grid indeces\n

    Parameters
    ----------
    point : tuple (required)
        Point to be converted to indeces
    xls : numpy array (required)
        Array describing the model geometry
    """
    # index of best fitting x point in L1 sense
    xind = [abs(point[num2] - xls[num2]).argmin() for num2 in range(len(xls))]
    return tuple(xind)


def ind2coord(point, xls):
    """
    Convert the grid indeces of a point to spatial coordinates\n

    Parameters
    ----------
    point : tuple (required)
        Point to be converted to indeces
    xls : numpy array (required)
        Array describing the model geometry
    """
    raise Exception("Not yet written")


# --------------- Functions for building/reading grids --------------- #
def load_grid(base_name, gtype=None):
    """
    Function for reading a binary binary grid

    Parameters
    ----------
    base_name : str (required)
        Name of the grid file (minus extension)
    gtype : str (default=None)
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
    if gtype == "PROB_DENSITY":
        binary_name = base_name + ".scat"
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


def read_header(path, gtype=None):
    """
    Function for reading NonLinLoc binary grid header files

    Parameters
    ----------
    path : str (required)
        Path to the header file
    gtype : str (optional, default=None)
        If specified, verify the grid is the expected format

    Returns
    -------
    header_data : dict
        Contents of the header file
    """
    with open(path, "r") as headfile:
        header = headfile.read().split()
    if not header[9] == gtype:
        raise TypeError(f"Grid format does not match specified grid type: {gtype}")
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


def read_bin_grid(path, grid):
    """
    Function for reading in a NonLinLoc binary grid

    Parameters
    ----------
    path : str (required)
        Path to the binary buffer
    grid : nllpy.grid Grid object (required)
        Grid object to dump the values into

    Returns
    -------
    grid : nllpy.grid Grid
        Grid as read from the file.
    """
    values = np.fromfile(path, dtype=np.float32)
    values = np.flip(values.reshape(grid.header["num_gps"]), axis=2)
    grid.values = values
    return grid


# --------------- Functions for building/reading grids --------------- #


def write_header(path, gps, origin, spacing, gtype):
    """
    Function for writing a NonLinLoc binary grid header

    Parameters
    ----------
    path : str (required)
        Path for the header file
    gps : list-like (required)
        Number of grid points in the X, Y, and Z directions
    origin : list-like (required)
        X, Y, and Z coordinates of the grid origin (Z is positive
        up). Coordinates must be in the NonLinLoc coordinate system
        and must be in kilometers.
    spacing : float (required)
        Size of the grid cells in kilometers. Grid cells must be
        square.
    gtype : str (required)
        Type of grid being written. Valid options are: "VELOCITY",
        "SLOWNESS", "VEL2", "SLOW2", "SLOW_LEN", "TIME", "TIME2D",
        "PROB_DENSITY", "MISFIT", "ANGLE", "ANGLE2D"
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


def write_bin_grid(path, grid):
    """
    Function for writing a NonLinLoc binary grid

    Parameters
    ----------
    path : str (required)
        Path for the binary file
    grid : numpy array (required)
        Three-dimensional numpy array containing the values for the grid
    """
    grid = np.flip(grid, axis=2).reshape((1, grid.size))
    grid.astype("float32").tofile(path)


# ---------- Utilities for manipulating the values of a grid --------- #
def apply_layers(grid, layers):
    """
        Internal method for applying velocities from a 1D layered model

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
        if not isinstance(elev, Iterable) or isinstance(elev, str):
            raise TypeError(
                "velocity input should be a 2D list of values and elevations"
            )
        grid.values[:, :, grid.grid_points[2] <= elev[1]] = elev[0]


def apply_rectangles(
    grid, rectangles, tol=1e-6, conversion=None, conversion_kwargs=None
):
    """
    Internal method for perturbing grid values in rectangular grid regions

    Parameters
    ----------
    grid : Grid
        Grid on which to apply the layers
    rectangles : pandas DataFrame (required)
        List of velocity perturbations to apply. Required columns
        include ["DELTA", "XMIN", "XMAX", "YMIN", "YMAX"] where "DELTA" is
        some percentage of the current grid value
    tol : float (default=1e-6)
        Value to add to rectangle coordinates to prevent rounding
        errors. This value should be small relative to the grid
        spacing.
    conversion : list-like (default=None)
        See convert_coords
    conversion_kwargs : dict (default=None)
        See convert_coords
    """
    if isinstance(rectangles, str):
        path = rectangles
        if not os.path.isfile(path):
            raise OSError(f"rectangles file does not exist: {path}")
        rectangles = read_file(path)
        cols = {"DELTA", "XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"}
        if not cols.issubset(rectangles.columns):
            raise IOError(f"{path} is not a valid rectangles file")
    elif isinstance(rectangles, pd.DataFrame):
        pass
    else:
        raise TypeError("rectangles must be a pandas DataFrame")
    if conversion:
        temp_points = pd.DataFrame(columns=["X", "Y", "Z"])
        i = 0
        for num, r in rectangles.iterrows():
            temp_points.loc[i] = [r.XMIN, r.YMIN, r.ZMIN]
            temp_points.loc[i + 1] = [r.XMAX, r.YMAX, r.ZMAX]
            i = i + 2
        temp_points = convert_coords(
            temp_points,
            conversion=conversion,
            conversion_kwargs=conversion_kwargs,
            xout="X",
            yout="Y",
            zout="Z",
        )
        for num, r in temp_points.iterrows():  # Why did I do this this way?
            i = num // 2
            ind = rectangles.iloc[i].name
            if (num % 2) == 0:
                rectangles.loc[ind, "XMIN"] = r.X
                rectangles.loc[ind, "YMIN"] = r.Y
                rectangles.loc[ind, "ZMIN"] = r.Z
            else:
                rectangles.loc[ind, "XMAX"] = r.X
                rectangles.loc[ind, "YMAX"] = r.Y
                rectangles.loc[ind, "ZMAX"] = r.Z
    v = grid.values
    gmap = grid.grid_map
    for num, zone in rectangles.iterrows():
        delta = 0.01 * zone.DELTA
        # Use a ridiculous one-liner to modify the zones within the
        # defined rectangle
        v[
            (gmap[0] >= zone.XMIN - tol)
            & (gmap[0] <= zone.XMAX + tol)
            & (gmap[1] >= zone.YMIN - tol)
            & (gmap[1] <= zone.YMAX + tol)
            & (gmap[2] >= zone.ZMIN - tol)
            & (gmap[2] <= zone.ZMAX + tol)
        ] = (
            v[
                (gmap[0] >= zone.XMIN - tol)
                & (gmap[0] <= zone.XMAX + tol)
                & (gmap[1] >= zone.YMIN - tol)
                & (gmap[1] <= zone.YMAX + tol)
                & (gmap[2] >= zone.ZMIN - tol)
                & (gmap[2] <= zone.ZMAX + tol)
            ]
            * delta
        )
    return v


def apply_topo(
    grid,
    topo_points,
    air=0.343,
    method="nearest",
    conversion=None,
    conversion_kwargs=None,
    tolerance=1e-6,
    buffer=0,
    topo_label="TOPO",
):
    """
        Method for applying a topographic surface to the model

        Parameters
        ----------
        grid : Grid
            Grid on which to apply the layers
        topo_points : required
            Input containing the topography data
        air : float (default=0.343)
            Value to assign to the "air" blocks (blocks above the topography)
        method : str (default="nearest")
            Method used by scipy's griddata to interpolate the topography
            grid. Acceptable values include: "nearest", "linear", and
            "cubic"
        conversion : list-like (default=None)
            Coordinate conversion to apply to the points before mapping
            them to the velocity grid. Two-dimensional list-like option
            that specifies the various steps in the conversion process.
            Values should be of the form ["keyword", value]. Acceptable
            keywords are: "scale", "translate_x", "translate_y",
            "translate_z", "rotate_xy", "rotate_xz", "rotate_yz", and
            "project". See nllpy.util.convert_coords for more detail.
        conversion_kwargs : dict (default=None)
            If the conversion is a callable, these kwargs will get passed to it.
        tolerance : float (optional, default=1e-6)
            Should be small relative to the grid size. Deals with those
            pesky rounding errors.
        buffer : int (default=0)
            Number of cells above the topography to extend the "rock" values
            (i.e., the values that are below the topography)

        Notes
        -----
        If the input is a CSV file or pandas DataFrame, the following
        columns are required: ["X", "Y", "Z"]. If the input is a dxf file,
        the dxf should not contain any data other than the topography (in
        the form of LWPOLYLINEs, LINEs, POLYLINES, POINTS, and/or 3DFACEs)
        and must have the elevation data stored in the entities (Z
        coordinates shouldn't be 0.0).\n
        It should be noted that bizarre results may occur if the topo
        data does not extend beyond the grid region.\n
        """
    # Do some basic error checking and read in the topo data
    if isinstance(topo_points, pd.DataFrame):
        topo = topo_points
        if not {"X", "Y", "Z"}.issubset(topo.columns):
            raise KeyError(
                f"topo_points must contain the following columns: ['X', 'Y', 'Z']"
            )
    elif isinstance(topo_points, str):
        if not os.path.isfile(topo_points):
            raise OSError(f"topo file does not exist: {topo_points}")
        topo = read_file(topo_points, funcs=(pd.read_csv, _read_topo_dxf))
        if not {"X", "Y", "Z"}.issubset(topo.columns):
            raise IOError(f"{topo_points} is not a valid topo file")
    else:
        raise TypeError("An invalid topo_points was provided to apply_topo")
    if not {"X", "Y", "Z"}.issubset(topo.columns):
        raise KeyError(
            f"topo_points must contain the following columns: ['X', 'Y', 'Z']"
        )

    if method not in ["nearest", "linear", "cubic"]:
        raise ValueError(f"Unsupported interpolation format: {method}")

    # Apply a coordinate conversion, if necessary
    if conversion:
        topo = convert_coords(
            topo,
            conversion,
            xout="X",
            yout="Y",
            zout="Z",
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
        np.array(topo[["X", "Y"]]),
        np.array(topo["Z"]),
        (grid_x, grid_y),
        method="nearest",
    )
    if method == "nearest":
        topo_grid = nearest_grid
    else:
        topo_grid = griddata(
            np.array(topo[["X", "Y"]]),
            np.array(topo["Z"]),
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
    # Want to add an extra grid cell above the topo layer because of
    # how Grid2Time interpolates the velocity between gps
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
def plt_grid(x1_coords, x2_coords, values, **kwargs):
    """
    Function to plot a colormap of a grid

    Parameters
    ----------
    x1_coords : array-like (required)
        Coordinates of grid points along the horizontal axis of the plot
    x2_coords : array-like (required)
        Coordinates of grid points along the vertical axis of the plot
    values : numpy array (required)
        Values stored in the grid

    Kwargs
    ------
    figsize : tuple (default=(9,9))
        Size of the figure
    cmap : str or matplotlib Colormap (or related) (default="rainbow")
        Colormap to use for displaying grid values
    alpha : float (default=0.5)
        Transparency value for the colormap
    legend_label : str (default="Value")
        Label to display on the colorbar
    shading : str (default="flat")
        Indicates whether to use shading when rendering. Acceptable
        values are "flat" for a solid color in each grid cell or
        "gouraud" to apply Gouraud shading (see matplotlib docs)

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
        2, 2, width_ratios=[10, 0.5], height_ratios=[10, 0.5], wspace=0.05, hspace=0.05
    )
    ax = fig.add_subplot(gs[0])

    # Plot the grid with the appropriate user args
    cmesh = ax.pcolormesh(
        x1_coords, x2_coords, values, cmap=cmap, alpha=alpha, shading=shading
    )

    # Make the plot look nice and add a legend
    ax.xaxis.tick_top()

    ax2 = fig.add_subplot(gs[2])
    fig.colorbar(cmesh, cax=ax2, orientation="horizontal", label=legend_label)

    # Make the plot look nice
    ax.set_aspect("equal")
    return fig, ax


def grid_cross(grid, coord, direction="X"):
    """
    Function for returning a profile from a two-dimensional grid

    Parameters
    ----------
    grid : Grid
        Grid from which to pull the values
    coord : float
        Coordinate along which to get the profile
    direction : str
        Direction along which to get the profile. Possible values are "X"
        for a profile that parallel to the x-axis and "Y" for a profile
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


# ------------------ Coordinate conversion utilities ----------------- #
def convert_coords(
    points,
    conversion,
    xcol="X",
    ycol="Y",
    zcol="Z",
    xout="X_GRID",
    yout="Y_GRID",
    zout="Z_GRID",
    conversion_kwargs=None,
    inplace=False,
):
    """
    Converts coordinates from one system to another

    Parameters
    ----------
    points : DataFrame (required)
        DataFrame containing the coordinates of the points to be
        converted. The following columns are required: ["X", "Y",
        "Z"(optional)]
    conversion : list-like or callable (required)
        Two-dimensional list-like option that specifies the various steps
        in the conversion process. Values should be of the form
        ["keyword", value]. Acceptable keywords are: "scale",
        "translate_x", "translate_y", "translate_z", "rotate_xy",
        "rotate_xz", "rotate_yz", and "project". Alternatively, a callable
        that accepts and returns a dataframe can be specified with optional
        kwargs to handle the conversion.
    xcol : str (default="X")
        Name of the input x-coordinate column
    ycol : str (default="Y")
        Name of the input y-coordinate column
    zcol : str (default="Z")
        Name of the input z-coordinate column
    xout : str (default="X_GRID")
        Name of the output x-coordinate column
    yout : str (default="Y_GRID")
        Name of the output y-coordinate column
    zout : str (default="Z_GRID")
        Name of the output z-coordinate column
    conversion_kwargs : dict (default=None)
        Used if conversion is a callable. kwargs to be passed to the callable.
    inplace : bool (default=False)
        Indicates whether the coordinate conversion should be done on the
        DataFrame in place or o new DataFrame should be returned.

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
        Strings for the pyproj.Proj class projection (ex:
        "+init=EPSG:32611"). Keys in the dictionary should be "from" and
        "to"\n

    Returns
    -------
    pandas Dataframe :
        Dataframe (either a copy or the original) with the converted coordinates
    """
    valid_keys = [
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
    ]
    # Copy the input if not inplace
    if not inplace:
        points = copy(points)

    # Define the columns to do the conversion on
    points[xout] = points[xcol]
    points[yout] = points[ycol]
    points[zout] = points[zcol]

    # Deal with a callable conversion
    if isinstance(conversion, Callable):
        try:
            if conversion_kwargs is not None:
                points = conversion(points, **conversion_kwargs)
            else:
                points = conversion(points)
        except Exception as e:
            raise RuntimeError(f"Conversion callable raised an exception: {e}")
        if not isinstance(points, pd.DataFrame):
            raise ValueError("Callable conversion must return a pandas DataFrame")
        return points

    # Do some checking to make sure the conversion is kosher
    if not isinstance(conversion, Iterable):
        raise TypeError(f"conversion should be a list of transformations")
    try:
        if not conversion[0][0].lower() in valid_keys:
            raise TypeError(f"conversion should be a list of transformations")
    except AttributeError:
        raise TypeError(f"conversion should be a list of transformations")

    # Go through each stage of the conversion
    for stage in conversion:
        # Make sure the stage is valid
        key = stage[0].lower()
        value = stage[1]
        if key not in valid_keys:
            raise ValueError(f"Invalid conversion operation: {key}")
        elif not key == "project":
            try:
                value = float(value)
            except ValueError:
                raise TypeError(f"value must be a float for key: {key}, {value}")
        # Do the conversion
        if key == "scale_x":
            points[xout] = points[xout] * value
        elif key == "scale_y":
            points[yout] = points[yout] * value
        elif key == "scale_z":
            points[zout] = points[zout] * value
        elif key == "translate_x":
            points[xout] = points[xout] + value
        elif key == "translate_y":
            points[yout] = points[yout] + value
        elif key == "translate_z":
            points[zout] = points[zout] + value
        elif key == "rotate_xy":
            points[xout], points[yout] = rotate_point(points[xout], points[yout], value)
        elif key == "rotate_xz":
            points[xout], points[zout] = rotate_point(points[xout], points[zout], value)
        elif key == "rotate_yz":
            points[yout], points[zout] = rotate_point(points[yout], points[zout], value)
        elif key == "project":
            try:
                try:
                    fro = pyproj.Proj(str(value["from"]))
                except RuntimeError:
                    raise ValueError(f"{value['from']} is not a valid projection")
                try:
                    to = pyproj.Proj(str(value["to"]))
                except RuntimeError:
                    raise ValueError(f"{value['to']} is not a valid projection")
                points[xout], points[yout] = pyproj.transform(
                    fro, to, points[xout].tolist(), points[yout].tolist()
                )
            except NameError:
                raise ImportError("pyproj is not installed on this system")
    return points


def rotate_point(x, y, ang):  # around origin
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


# ------------- Internal functions for handling dxfs ----------------- #
def _read_topo_dxf(dxf, line_end="\n"):
    with open(dxf, "r") as f:
        parsed = f.read()
    # Split by section
    parsed = parsed.split(f"  0{line_end}SECTION{line_end}")
    # Get the entities and split them up
    try:
        parsed = parsed[5].split(
            f"{line_end}  0{line_end}"
        )  # Technically dangerous... should really explicitly seek out the entities section
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
            entity["Z"] = elev
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
    # Parse into a key, value DataFrame
    e = np.array(entity[1:])
    if not (len(e) % 2) == 0:
        raise IOError(f"Corrupt dxf file while parsing {entity[0]}")
    e = e.reshape((len(e) // 2, 2))
    return pd.DataFrame(e, columns=["CODE", "VALUE"], dtype="object")


def _parse_pointlike(entity):
    entity = _entity_boilerplate(entity)
    # Retrieve the X, Y, and Z coordinates of the point
    inlist = [" 10", " 20", " 30"]
    return _reshape_points(entity, inlist)


def _reshape_points(df, inlist, use_z=True):
    # Drop all of the extraneous stuff
    points = df.loc[df.CODE.isin(inlist)]
    # Reshape the series as a table of X, Y, and Z (optional) coordinates
    points = np.array(points.VALUE, dtype=np.float64)
    if use_z:
        dim = 3
        cols = ["X", "Y", "Z"]
    else:
        dim = 2
        cols = ["X", "Y"]
    if not (len(points) % dim) == 0:
        handle = df.loc[df.CODE == "  5"].iloc[0].VALUE
        raise IOError(f"Corrupt dxf file crashed while parsing entity: {handle}")
    return pd.DataFrame(points.reshape((len(points) // dim, dim)), columns=cols)
