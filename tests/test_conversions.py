import os

import numpy as np
import pandas as pd
import pytest

import obsplus.conversions

wgs84 = "+init=EPSG:4326"  # This is going to have issues on newer versions of pyproj?
grid_coords = "+init=EPSG:2926"


def callable_conversion(arr, num):
    arr *= num
    return arr


def bogus_conversion_callable(x):
    x = "a"
    return x


def bogus_conversion_callable1(df):
    return df


def excepting_conversion_callable(x):
    raise ValueError("Hi")


conversions = {
    "test_conversion": [
        ("SCALE_X", 1 / 0.3048),
        ("SCALE_Y", 1 / 0.3048),
        ("SCALE_Z", 1 / 0.3048),
        ("TRANSLATE_X", 1),
        ("TRANSLATE_Y", 1),
        ("TRANSLATE_Z", 1),
        ("ROTATE_XY", 1),
        ("ROTATE_XZ", 1),
        ("ROTATE_YZ", 1),
    ],
    "test_project": [("PROJECT", {"from": wgs84, "to": grid_coords})],
    "convert_to_km": [
        ("SCALE_X", 0.3048 * 0.001),
        ("SCALE_Y", 0.3048 * 0.001),
        ("SCALE_Z", 0.3048 * 0.001),
    ],
    "convert_callable": callable_conversion,
    "bogus_callable": bogus_conversion_callable,
    "bogus_callable1": bogus_conversion_callable1,
    "raising_callable": excepting_conversion_callable,
}


class TestCoordinateConversion:
    """ Tests for making sure the coordinate conversions work as expected """

    # Fixture
    @pytest.fixture(scope="class")
    def points(self, grid_path):
        return pd.read_csv(os.path.join(grid_path, "test_points.csv"))

    # Tests
    def test_conversion(self, points):
        """Verify it is possible to convert coordinates (from a pandas dataframe)"""
        df = obsplus.conversions.convert_coords(
            points,
            conversion=conversions["test_conversion"],
            x_in="X",
            y_in="Y",
            z_in="Z",
        )
        assert not id(df) == id(points)
        assert {"x_conv", "y_conv", "z_conv"}.issubset(df.columns)
        assert "x_conv" not in points.columns
        point = df.iloc[0]
        assert np.isclose(point.x_conv, -5.265_527_623_648_098_1)
        assert np.isclose(point.y_conv, 13.367_252_189_813_943)
        assert np.isclose(point.z_conv, 39.972_122_227_570_381)

    def test_conversion_df_point(self, points):
        """ check the edge case of a single point being passed as a pandas DataFrame """
        df = obsplus.conversions.convert_coords(
            points.loc[0:1],
            conversion=conversions["test_conversion"],
            x_in="X",
            y_in="Y",
            z_in="Z",
        )
        point = df.iloc[0]
        assert np.isclose(point.x_conv, -5.265_527_623_648_098_1)
        assert np.isclose(point.y_conv, 13.367_252_189_813_943)
        assert np.isclose(point.z_conv, 39.972_122_227_570_381)

    def test_conversion_tuple(self):
        """Verify it is possible to convert coordinates (from a tuple)"""
        # Test a multiple-point tuple
        points = ((1, 2, 3), (4, 5, 6))
        points = obsplus.conversions.convert_coords(
            points, conversion=conversions["convert_to_km"]
        )
        assert isinstance(points, tuple)
        assert len(points) == 2
        assert np.isclose(points[0][0], 0.3048 * 0.001)

    def test_conversion_single_point_tuple(self):
        """ Verify it is possible to convert coordinates (from a tuple containing a single point) """
        point = (1, 2, 3)
        point = obsplus.conversions.convert_coords(
            point, conversion=conversions["convert_to_km"]
        )
        assert isinstance(point, tuple)
        assert len(point) == 3
        assert np.isclose(point[0], 0.3048 * 0.001)

    def test_conversion_bogus_tuple(self):
        """ Verify that an invalid tuple raises"""
        point = (1, 2)
        with pytest.raises(TypeError):
            obsplus.conversions.convert_coords(
                point, conversion=conversions["convert_to_km"]
            )

        point = ((1, 2, 3), (1, 2))
        with pytest.raises(TypeError):
            obsplus.conversions.convert_coords(
                point, conversion=conversions["convert_to_km"]
            )

    def test_conversion_list(self):
        """Verify it is possible to convert coordinates (from a list)"""
        # Test a multiple-point tuple
        points = [[1, 2, 3], [4, 5, 6]]
        points = obsplus.conversions.convert_coords(
            points, conversion=conversions["convert_to_km"]
        )
        assert isinstance(points, list)
        assert len(points) == 2
        assert np.isclose(points[0][0], 0.3048 * 0.001)

    def test_conversion_single_point_list(self):
        """ Verify it is possible to convert coordinates (from a list containing a single point) """
        point = [1, 2, 3]
        point = obsplus.conversions.convert_coords(
            point, conversion=conversions["convert_to_km"]
        )
        assert isinstance(point, list)
        assert len(point) == 3
        assert np.isclose(point[0], 0.3048 * 0.001)

    def test_conversion_bogus_list(self):
        """ Verify that an invalid list raises"""
        point = [1, 2]
        with pytest.raises(TypeError):
            obsplus.conversions.convert_coords(
                point, conversion=conversions["convert_to_km"]
            )

        point = [[1, 2, 3], [1, 2]]
        with pytest.raises(TypeError):
            obsplus.conversions.convert_coords(
                point, conversion=conversions["convert_to_km"]
            )

    def test_conversion_array(self, points):
        points = np.array(points[["X", "Y", "Z"]])
        points = obsplus.conversions.convert_coords(
            points, conversion=conversions["test_conversion"]
        )
        point = points[0]
        assert np.isclose(point[0], -5.265_527_623_648_098_1)
        assert np.isclose(point[1], 13.367_252_189_813_943)
        assert np.isclose(point[2], 39.972_122_227_570_381)

    def test_conversion_project(self, points):
        """Verify it is possible to convert station coords to grid coords"""
        try:
            import pyproj
        except ModuleNotFoundError:
            pytest.skip("pyproj is not installed on this machine. Skipping.")
        df = obsplus.conversions.convert_coords(
            points, conversion=conversions["test_project"], x_in="X", y_in="Y", z_in="Z"
        )
        point = df.iloc[0]
        assert np.isclose(point.x_conv, 11_337_944.568_914_454)
        assert np.isclose(point.y_conv, 7_426_524.761_504_985_4)
        assert np.isclose(point.z_conv, 2.75)

    def test_bogus_conversion(self, points):
        """Verify a bogus conversion raises"""
        with pytest.raises(TypeError):
            obsplus.conversions.convert_coords(
                points, conversion="a", x_in="X", y_in="Y", z_in="Z"
            )

    def test_conversion_callable(self, points):
        """Verify a callable conversion works"""
        df = obsplus.conversions.convert_coords(
            points,
            conversion=conversions["convert_callable"],
            conversion_kwargs={"num": 0.3048},
            x_in="X",
            y_in="Y",
            z_in="Z",
        )
        point = df.iloc[0]
        assert np.isclose(point.x_conv, 3.395_472)
        assert np.isclose(point.y_conv, 1.499_616)
        assert np.isclose(point.z_conv, 0.8382)

    def test_bogus_conversion_callable(self, points):
        """Verify a bogus callable conversion raises"""
        with pytest.raises(TypeError):
            obsplus.conversions.convert_coords(
                points,
                conversion=conversions["bogus_callable"],
                x_in="X",
                y_in="Y",
                z_in="Z",
            )

    def test_excepting_conversion_callable(self, points):
        """Verify that a callable that excepts raises in a predictable manner"""
        with pytest.raises(RuntimeError):
            obsplus.conversions.convert_coords(
                points,
                conversion=conversions["raising_callable"],
                x_in="X",
                y_in="Y",
                z_in="Z",
            )
