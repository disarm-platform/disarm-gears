import pytest
import numpy as np
import pandas as pd
import geopandas
from shapely import geometry
from disarm_gears.frames import TilePattern
from disarm_gears.util.buffers import voronoi_polygons


# Inputs
n_points = 10
g_points = np.random.uniform(0, 1, n_points * 2).reshape(n_points, -1)
vor = voronoi_polygons(g_points)
b_attrib = np.random.random(25)
g_attrib_1 = np.random.random(10)
g_attrib_2 = np.random.random(40).reshape(10, -1)
g_attrib_3 = pd.DataFrame({li: ci for li,ci in zip(['a', 'b', 'c', 'd'], g_attrib_2.T)})
X = np.vstack([g_points.copy()[5:], np.array([10, 10])])
B = geopandas.GeoDataFrame({'id': [0], 'geometry': [geometry.Polygon(((0.2, 0.3), (0.2, 0.8),
                                                                      (0.7, 0.8), (0.2, 0.3)))]})
B2 = geopandas.GeoDataFrame({'id': [0, 1], 'geometry': [geometry.Polygon(((0.2, 0.3), (0.2, 0.8),
                                                                          (0.7, 0.8), (0.2, 0.3))),
                                                        geometry.Polygon(((0.2, 0.3), (0.7, 0.3),
                                                                          (0.7, 0.8), (0.2, 0.3)))]})

# Demo object used repeatedly
sf_0 = TilePattern(geometries=vor.geometry, attributes=None, crs=None)

def test_inputs():
    # Check bad inputs
    with pytest.raises(AssertionError):
        TilePattern(geometries=g_points)
    with pytest.raises(NotImplementedError):
        TilePattern(geometries=vor.geometry, attributes=None, crs=0)

def test_outputs():

    # Check output types
    assert isinstance(sf_0.region, geopandas.GeoDataFrame)
    assert isinstance(sf_0.centroids, np.ndarray)

    # Check sf.region shape
    sf_0.region.ndim == 2
    sf_0.region.shape[0] == n_points
    sf_0.region.shape[1] == 1

    # Check sf.region.columns
    'geometry' in sf_0.region.columns

    # Check box type
    assert isinstance(sf_0.box, pd.DataFrame)
    sf_0.box.ndim == 2
    sf_0.box.shape[0] == 2
    sf_0.box.shape[1] == 2

def test_attributes_array():

    assert sf_0.attributes_array() is None

    _attr = np.random.uniform(0, 100, g_points.size).reshape(-1, 2)
    sf_1 = TilePattern(geometries=vor.geometry, attributes=_attr, crs=None)
    sf_attr = sf_1.attributes_array()
    assert isinstance(sf_attr, np.ndarray)
    assert sf_attr.shape[0] == _attr.shape[0]
    assert sf_attr.shape[1] == _attr.shape[1]

def test_set_boundary():

    sf_1 = TilePattern(geometries=vor.geometry, attributes=None, crs=None)
    with pytest.raises(AssertionError):
        sf_1.set_boundary(B=B.geometry[0])
    sf_1.set_boundary(B=B)
    assert isinstance(sf_1.boundary, geopandas.GeoDataFrame)

    sf_2 = TilePattern(geometries=vor.geometry, attributes=None, crs=None)
    sf_2.set_boundary(B2)
    assert sf_1.region.shape[0] < sf_2.region.shape[0]

def test_locate():

    ix = sf_0.locate(X=X)
    assert isinstance(ix, np.ndarray)
    assert ix.ndim == 1
    assert ix.size == X.shape[0]
    assert ix[-1] == -1
    assert np.all(ix[:-1] - np.arange(5, 10) == 0)


def test_points_to_frame():

    nx = X.shape[0]
    summary = sf_0.points_to_frame(X=X, group_by=None)
    assert isinstance(summary, pd.DataFrame)
    assert summary.ndim == 2
    assert summary.shape[0] == nx - 1
    assert summary['var_0'].sum() == nx - 1
    assert np.unique(summary['tile']).size == nx - 1


    with pytest.raises(AssertionError):
        sf_0.points_to_frame(X=np.vstack([X] * 3),
                             group_by=[['A'] * 2 * nx + ['B'] * nx])

    summary = sf_0.points_to_frame(X=np.vstack([X] * 3),
                                   group_by=np.hstack([['A'] * 2 * nx + ['B'] * nx]))
    assert summary.shape[0] == 2 * (nx - 1)
    assert 'var_0' in summary.columns
    assert np.all(summary.loc[summary.group == 'A', 'var_0'] == 2)
    assert np.all(summary.loc[summary.group == 'B', 'var_0'] == 1)


def test_marked_points_to_frame():


    nx = X.shape[0]
    summary1 = sf_0.marked_points_to_frame(X=np.vstack([X] * 3),
                                           Y=np.hstack([[10] * nx + [20] * nx + [30] * nx]),
                                           fun='sum')
    summary2 = sf_0.marked_points_to_frame(X=np.vstack([X] * 3),
                                           Y=np.hstack([[10] * nx + [20] * nx + [30] * nx]),
                                           fun='mean')
    summary3 = sf_0.marked_points_to_frame(X=np.vstack([X] * 3),
                                           Y=np.hstack([[10] * nx + [20] * nx + [30] * nx]),
                                           group_by=np.hstack([['A'] * 2 * nx + ['B'] * nx]),
                                           fun='mean')

    assert np.all(summary1['var_0'] == 60)
    assert np.all(summary2['var_0'] == 20)
    assert np.all(summary3.loc[summary3.group == 'A', 'var_0'] == 15)
    assert np.all(summary3.loc[summary3.group == 'B', 'var_0'] == 30)

    with pytest.raises(NotImplementedError):
        sf_0.marked_points_to_frame(X=X, Y=np.array([10] * nx), fun='var')
