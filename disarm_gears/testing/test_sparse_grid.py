import pytest
import numpy as np
import pandas as pd
import geopandas
from shapely import geometry
from disarm_gears.frames import SparseGrid

# Inputs
X1 = np.array([[2, 0], [0, 1], [1, 1], [1, 2]]) + .5


#def test_inputs():
#    # Check bad inputs
#    with pytest.raises(AssertionError):
#        SparseGrid(x_lim=(0, 3), y_lim=(0, 3), n_cols=3, n_rows=3, tag_prefix='')
#    with pytest.raises(AssertionError):
#        PointPattern(points=b_points_1)
#    with pytest.raises(AssertionError):
#        PointPattern(points=b_points_2)
#    with pytest.raises(AssertionError):
#        PointPattern(points=g_points, attributes=b_attrib)
#    with pytest.raises(NotImplementedError):
#        PointPattern(points=g_points, attributes=None, crs=0)
#    with pytest.raises(ValueError):
#        PointPattern(points=g_points, attributes=list())
#    with pytest.raises(ValueError):
#        PointPattern(points=g_points, attributes=b_attrib.reshape(1, 1, -1))


def test_outputs():
    # Check output types
    sg = SparseGrid(x_lim=(0, 3), y_lim=(0, 3), n_cols=3, n_rows=3, tag_prefix='')
    sg.dx == 1
    sg.dy == 1
    len(sg.tags) == 9
    isinstance(sg, geopandas.GeoDataFrame)

def test_add_polygon_from_xy():
    # Check that grids are correctly identified
    sg = SparseGrid(x_lim=(0, 3), y_lim=(0, 3), n_cols=3, n_rows=3, tag_prefix='')
    sg.add_polygon_from_xy(X1)
    sg.sparse_frame.shape[0] == 4
    isinstance(sg.sparse_frame.geometry[0], geometry.Polygon)
    isinstance(sg.sparse_frame.geometry[1], geometry.Polygon)
    isinstance(sg.sparse_frame.geometry[2], geometry.Polygon)
    isinstance(sg.sparse_frame.geometry[3], geometry.Polygon)


def test_get_simplified():
    # Check that grids are correctly identified
    sg = SparseGrid(x_lim=(0, 3), y_lim=(0, 3), n_cols=3, n_rows=3, tag_prefix='')
    sg.add_polygon_from_xy(X1)
    sgeop = sg.get_simplified()
    isinstance(sgeop, geopandas.GeoDataFrame)
    sgeop.shape[0] == 2


