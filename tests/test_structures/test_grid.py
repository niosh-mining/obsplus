#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:46:43 2017

@author: shawn
"""

import pytest
import os
from copy import deepcopy
import tempfile
import shutil

import numpy as np
import pandas as pd

import obsplus.structures.grid as obsgrid
from obsplus.structures.grid import Grid


# Stuff for coordinate conversions
from tests.test_conversions import conversions


# --- Functions for tests
def check_grid_bounds(grid, origin, spacing, num_gps):
    for num, val in enumerate(origin):
        space = np.linspace(val, val + spacing[num] * (num_gps[num] - 1), num_gps[num])
        np.testing.assert_array_almost_equal(grid.grid_points[num], space)
        np.testing.assert_array_almost_equal(num_gps, grid.grid_map[num].shape)
        np.testing.assert_almost_equal(val, grid.grid_map[num].min())
        np.testing.assert_almost_equal(
            val + spacing[num] * (num_gps[num] - 1), grid.grid_map[num].max()
        )


# --- Set up a temporary directory
@pytest.fixture(scope="class")
def temp_dir():
    """ create a temporary working directory """
    with tempfile.TemporaryDirectory() as td:
        yield td
    if os.path.exists(td):
        shutil.rmtree(td)


# --- Define common inputs
@pytest.fixture(scope="class")
def inputs(temp_dir):
    """ common inputs for creating a velocity model grid """
    inp = {
        "base_name": os.path.join(temp_dir, "test_mod"),
        "origin": [9.14, 3.67, 0.75],
        "spacing": [0.05, 0.05, 0.05],
        "num_gps": [81, 57, 41],
        "num_cells": [80, 56, 40],
        "vel_input": [[2.71, 3.00]],
    }
    # "vtype": "1d"}
    return inp


# --- Grids for tests
@pytest.fixture(scope="class")
def velocity_model(inputs):
    """ Create simple 3D velocity model using a Grid object """
    vel = Grid(
        base_name=inputs["base_name"],
        gtype="VELOCITY",
        origin=inputs["origin"],
        spacing=inputs["spacing"],
        num_gps=inputs["num_gps"],
    )
    obsgrid.apply_layers(vel, inputs["vel_input"])
    return vel


@pytest.fixture(scope="class")
def velocity_model_num_cells(inputs):
    vel = Grid(
        base_name=inputs["base_name"],
        gtype="VELOCITY",
        origin=inputs["origin"],
        spacing=inputs["spacing"],
        num_cells=inputs["num_cells"],
    )
    obsgrid.apply_layers(vel, inputs["vel_input"])
    return vel


class TestGrid:
    """
    Tests to verify the Grid structure can be built correctly
    """

    # --- Functions
    def raising_grid(self, params, exception):
        with pytest.raises(exception):
            Grid(**params)

    # --- Tests
    def test_gp_velmod(self, velocity_model, inputs):
        """Make sure the bounds and values of a velocity model are correct when defining using number of grid points"""
        check_grid_bounds(
            velocity_model, inputs["origin"], inputs["spacing"], inputs["num_gps"]
        )

    def test_cell_velmod(self, velocity_model_num_cells, inputs):
        """Make sure the bounds and values of a velocity model are correct when defining using number of grid points"""
        check_grid_bounds(
            velocity_model_num_cells,
            inputs["origin"],
            inputs["spacing"],
            inputs["num_gps"],
        )

    def test_bogus_gridspec(self, inputs):
        """Verify bogus grid dimensions raise"""
        inputs = deepcopy(inputs)
        inputs.pop("vel_input")
        for spec in ["origin", "spacing", "num_gps", "num_cells"]:
            # Make sure an invalid spec raises
            inps = deepcopy(inputs)
            inps[spec] = "abc"
            self.raising_grid(inps, TypeError)
            # Make sure an invalid item in a spec raises
            inps = deepcopy(inputs)
            if (
                spec == "num_cells"
            ):  # Don't remember the reasoning for handling num_cells separately
                inps[spec] = ["abc", 2, 3]
            else:
                inps[spec][0] = "abc"
            self.raising_grid(inps, TypeError)
            # Make sure an extra param in a spec raises
            inps = deepcopy(inputs)
            if spec == "num_cells":
                inps[spec] = [1, 2, 3, 4]
            else:
                inps[spec] = inps[spec] + [1]
            self.raising_grid(inps, ValueError)

    def test_mismatched_num_gps_cells(self, inputs):
        """Verify a bogus origin raises"""
        inps = deepcopy(inputs)
        inps.pop("vel_input")
        inps["num_gps"] = [10, 20, 30]
        inps["num_cells"] = [20, 30, 40]
        self.raising_grid(inps, ValueError)


class TestGridReadWrite:
    """ Tests for verifying that it is possible to read/write grid objects """

    @pytest.fixture(scope="class")
    def write_model(self, velocity_model, inputs):
        more_complex_layers = [[2.71, 3.00], [3.2, 2.5], [4.0, 2.0]]
        vm = deepcopy(velocity_model)
        obsgrid.apply_layers(vm, more_complex_layers)
        vm.write()
        return vm

    @pytest.fixture(scope="class")
    def loaded_grid(self, write_model, inputs):
        return obsgrid.load_grid(inputs["base_name"], gtype="VELOCITY")

    def test_header(self, inputs, write_model):
        """ Make sure the header file is correct """
        check = (
            "81 57 41 9.140 3.670 -2.750 0.050 0.050 0.050 VELOCITY FLOAT\n"
            "TRANSFORM NONE\n"
        )
        with open(f"{inputs['base_name']}.hdr") as f:
            actual = f.read()
        assert actual == check

    def test_load_bounds(self, write_model, loaded_grid, inputs):
        """ Check the bounds of the newly loaded grid """
        check_grid_bounds(
            loaded_grid, inputs["origin"], inputs["spacing"], inputs["num_gps"]
        )

    def test_grid_values(self, write_model, loaded_grid):
        """ Make sure the components of the loaded grid match what was written """
        for ind, gm in enumerate(loaded_grid.grid_map):
            np.testing.assert_array_almost_equal(gm, write_model.grid_map[ind])
        np.testing.assert_array_almost_equal(loaded_grid.values, write_model.values)


class TestGridPlotting:
    """ Tests for verifying that grid plotting functions work correctly """

    def test_plot_slice(self, velocity_model):
        velocity_model.plot_slice(1.2)

    def test_plot_2d(self):
        origin = [0, 0]
        spacing = [1, 1]
        num_gps = [9, 11]
        grid = Grid(
            base_name="plane",
            gtype="UNKNOWN",
            origin=origin,
            spacing=spacing,
            num_gps=num_gps,
        )
        grid.plot_2d()


class TestManipulateGrids:
    """ Tests for verifying functions for manipulating the values of grids """

    # Tests for layered models
    def test_bogus_layers(self, velocity_model):
        """Verify a bogus layer specification"""
        # Totally invalid input, bad item in list, bad value in layer
        for mod in ["abcd", [[2.71, 3.00], "abcd"], [[2.71, "a"]]]:
            with pytest.raises(TypeError):
                obsgrid.apply_layers(velocity_model, mod)

    # Tests for perturbing rectangular model regions
    def test_perturb_rectangle(self, velocity_model, grid_path):
        """Verify that a rectangular region of a grid can be perturbed"""
        rectangle = os.path.join(grid_path, "test_lvz.csv")
        vm = deepcopy(velocity_model)
        obsgrid.apply_rectangles(vm, rectangle, conversion=conversions["convert_to_km"])

        x_coords = np.linspace(9.14, 13.14, 21)
        x_changed = np.extract(
            (np.greater_equal(x_coords, 10.03) & np.less_equal(x_coords, 11)), x_coords
        )
        x_unchanged = np.extract(
            (np.less(x_coords, 10.03) | np.greater(x_coords, 11)), x_coords
        )
        y_coords = np.linspace(3.67, 6.42, 56)
        y_changed = np.extract(
            (np.greater_equal(y_coords, 4.51) & np.less_equal(y_coords, 4.98)), y_coords
        )
        y_unchanged = np.extract(
            (np.less(y_coords, 4.51) | np.greater(y_coords, 4.98)), y_coords
        )
        z_coords = np.linspace(0.75, 2.75, 21)
        z_changed = np.extract(
            (np.greater_equal(z_coords, 1.19) & np.less_equal(z_coords, 1.41)), z_coords
        )
        z_unchanged = np.extract(
            (np.less(z_coords, 1.19) | np.greater(z_coords, 1.19)), z_coords
        )
        # Check that the velocity has been changed in the perturbed zone
        for x in x_changed:
            for y in y_changed:
                for z in z_changed:
                    val = vm.get_value((x, y, z), interpolate=False)
                    assert np.isclose(val, 2.439)
        # Check that the rest of the model is unchanged
        for x in x_unchanged:
            for y in y_unchanged:
                for z in z_unchanged:
                    val = vm.get_value((x, y, z), interpolate=False)
                    assert np.isclose(val, 2.71)

    def test_bogus_rectangle(self, velocity_model, grid_path):
        rectangle = os.path.join(grid_path, "simple_topo.csv")
        with pytest.raises(IOError):
            obsgrid.apply_rectangles(velocity_model, rectangle)

    # Tests for incorporating topography
    def test_add_topo_csv(self, velocity_model, grid_path):
        """Verify that a csv file can be used to add topography"""
        vm = deepcopy(velocity_model)
        topo = os.path.join(grid_path, "simple_topo.csv")
        obsgrid.apply_topo(vm, topo, method="linear")
        # Spot check the velocities
        # (need to be careful here... it could just be a limitation of the
        # interpolation method, not necessarily the grid...)
        checkpoints = pd.read_csv(os.path.join(grid_path, "topo_check_vals.csv"))
        for num, point in checkpoints.iterrows():
            val = vm.get_value((point.X, point.Y, point.Z), interpolate=False)
            assert np.isclose(val, 0.343) == point.AIR

    def test_add_topo_dxf(self, velocity_model, grid_path):
        """Verify that a dxf file can be used to add topography"""
        df = pd.read_csv(os.path.join(grid_path, "dxf_check.txt"))
        dxf_file = os.path.join(grid_path, "topo.dxf")
        vm = deepcopy(velocity_model)
        csv = obsgrid.apply_topo(
            vm, df, method="nearest", conversion=conversions["convert_to_km"]
        )
        dxf = obsgrid.apply_topo(
            vm, dxf_file, method="nearest", conversion=conversions["convert_to_km"]
        )
        np.testing.assert_array_almost_equal(csv.values, dxf.values)

    def test_bogus_file(self, velocity_model, grid_path):
        topo = os.path.join(grid_path, "bogus.dxf")
        with pytest.raises(IOError):
            obsgrid.apply_topo(
                velocity_model,
                topo,
                method="nearest",
                conversion=conversions["convert_to_km"],
            )
        topo = os.path.join(grid_path, "test_modslow.P.mod.buf")
        with pytest.raises(IOError):
            obsgrid.apply_topo(
                velocity_model,
                topo,
                method="nearest",
                conversion=conversions["convert_to_km"],
            )


class TestValueRetrieval:
    """ Tests for retrieving data from grids """

    @pytest.fixture(scope="class")
    def topo_map(self, grid_path):
        """Make a very simple two-dimensional grid"""
        plane = pd.read_csv(os.path.join(grid_path, "plane.csv"))
        origin = [0, 0, 0]
        spacing = [1, 1, 1]
        num_gps = [9, 11, 1]
        grid = Grid(
            base_name="plane",
            gtype="UNKNOWN",
            origin=origin,
            spacing=spacing,
            num_gps=num_gps,
        )
        plane = obsgrid.apply_topo(grid, plane)
        return plane

    def test_get_x_profile(self, topo_map):
        """Pull a profile along the x-axis"""
        points, values = obsgrid.grid_cross(topo_map, 2, direction="X")
        assert len(points) == 9
        # The 'topo' map should increase linearly along the x-axis
        for ind, val in enumerate(values):
            if ind > 0:
                assert val > values[ind - 1]

    def test_get_y_profile(self, topo_map):
        """Pull a profile along the y-axis"""
        points, values = obsgrid.grid_cross(topo_map, 2, direction="Y")
        assert len(points) == 11
        # The 'topo' map should be constant along the y-axis
        assert np.isclose(values, values[0]).all()
