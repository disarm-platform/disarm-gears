import pytest
import numpy as np
import pandas as pd
import geopandas
from shapely import geometry
from disarm_gears.frames import PointPattern


# Inputs
b_points_1 = np.random.uniform(0, 1, 10)
b_points_2 = np.random.uniform(0, 1, 30).reshape(10, 3)
g_points = np.random.uniform(0, 1, 20).reshape(10, -1)
b_attrib = np.random.random(25)
g_attrib_1 = np.random.random(10)
g_attrib_2 = np.random.random(40).reshape(10, -1)
g_attrib_3 = pd.DataFrame({li: ci for li,ci in zip(['a', 'b', 'c', 'd'], g_attrib_2.T)})
n_points = g_points.shape[0]
X = np.vstack([g_points.copy()[5:], np.array([10, 10])])
B = geopandas.GeoDataFrame({'id': [0], 'geometry': [geometry.Polygon(((0.2, 0.3), (0.2, 0.8),
                                                                 (0.7, 0.8), (0.2, 0.3)))]})
B2 = geopandas.GeoDataFrame({'id': [0, 1], 'geometry': [geometry.Polygon(((0.2, 0.3), (0.2, 0.8),
                                                                     (0.7, 0.8), (0.2, 0.3))),
                                                        geometry.Polygon(((0.2, 0.3), (0.7, 0.3),
                                                                     (0.7, 0.8), (0.2, 0.3)))]})

def test_inputs():
    # Check bad inputs
    with pytest.raises(AssertionError):
        PointPattern(points=0)
    with pytest.raises(AssertionError):
        PointPattern(points=b_points_1)
    with pytest.raises(AssertionError):
        PointPattern(points=b_points_2)
    with pytest.raises(AssertionError):
        PointPattern(points=g_points, attributes=b_attrib)
    with pytest.raises(NotImplementedError):
        PointPattern(points=g_points, attributes=None, crs=0)
    with pytest.raises(ValueError):
        PointPattern(points=g_points, attributes=list())
    with pytest.raises(ValueError):
        PointPattern(points=g_points, attributes=b_attrib.reshape(1, 1, -1))


def test_outputs():
    # Check output types
    sf_0 = PointPattern(points=pd.DataFrame(g_points), attributes=None, crs=None)
    sf_1 = PointPattern(points=g_points, attributes=None, crs=None)
    sf_2 = PointPattern(points=g_points, attributes=g_attrib_1, crs=None)
    sf_3 = PointPattern(points=g_points, attributes=g_attrib_2, crs=None)
    sf_4 = PointPattern(points=g_points, attributes=g_attrib_3, crs=None)

    # Check sf.region is geopandas.GeoDataFrame
    isinstance(sf_2.centroids, np.ndarray)
    isinstance(sf_2.centroids, np.ndarray)
    sf_4.centroids.shape[0] == n_points
    isinstance(sf_1.region, geopandas.GeoDataFrame)
    isinstance(sf_0.region, geopandas.GeoDataFrame)
    isinstance(sf_3.region, geopandas.GeoDataFrame)

    # Check sf.region shape
    sf_1.region.ndim == 2
    sf_1.region.shape[0] == n_points
    sf_3.region.shape[0] == n_points
    sf_1.region.shape[1] == 1
    sf_2.region.shape[1] == 2
    sf_4.region.shape[1] == 5

    # Check sf.region.columns
    'geometry' in sf_3.region.columns
    'geometry' in sf_4.region.columns

    # Check attribute names
    np.array('var_%s' %i in sf_3.region.columns for i in range(4)).all()
    np.array(v in sf_3.region.columns for v in ['a', 'b', 'c', 'd']).all()

    # Check box type
    isinstance(sf_2.box, pd.DataFrame)
    sf_3.box.ndim == 2
    sf_1.box.shape[0] == 2
    sf_4.box.shape[1] == 2


def test_attributes_array():

    sf_1 = PointPattern(points=g_points, attributes=None, crs=None)
    sf_1.attributes_array() is None

    _attr = np.random.uniform(0, 100, g_points.size).reshape(-1, 2)
    sf_1 = PointPattern(points=g_points, attributes=_attr, crs=None)
    sf_attr = sf_1.attributes_array()
    isinstance(sf_attr, np.ndarray)
    sf_attr.shape[0] == _attr.shape[0]
    sf_attr.shape[1] == _attr.shape[1]


def test_set_boundary():

    new_points = np.array([[.22, .68, .68, .22], [.32, .32, .78, .78]]).T
    sf_1 = PointPattern(points=new_points, attributes=None, crs=None)
    with pytest.raises(AssertionError):
        sf_1.set_boundary(B=B.geometry[0])
    sf_1.set_boundary(B=B)
    assert isinstance(sf_1.boundary, geopandas.GeoDataFrame)
    assert sf_1.box.loc[0, 'x'] == 0.2
    assert sf_1.box.loc[1, 'x'] == 0.7
    assert sf_1.box.loc[0, 'y'] == 0.3
    assert sf_1.box.loc[1, 'y'] == 0.8

    sf_2 = PointPattern(points=new_points, attributes=None, crs=None)
    sf_2.set_boundary(B2)
    assert isinstance(sf_2.boundary, geopandas.GeoDataFrame)

    assert sf_1.region.shape[0] < sf_2.region.shape[0]


def test_make_grid():

    sf_0 = PointPattern(points=pd.DataFrame(g_points), attributes=None, crs=None)
    G0 = sf_0.make_grid(resolution=(8, 5))
    assert isinstance(G0, np.ndarray)
    assert G0.shape[0] == 8 * 5
    assert G0.shape[1] == 2

    sf_0.set_boundary(B)
    G1 = sf_0.make_grid(resolution=(8, 5), bounded=True)
    assert G0.shape[0] > G1.shape[0]


def test_make_attribute_series():

    sf_3 = PointPattern(points=g_points, attributes=g_attrib_3, crs=None)
    with pytest.raises(AssertionError):
        sf_3.make_attributes_series(knots=[0, 1])
    with pytest.raises(AssertionError):
        sf_3.make_attributes_series(knots=np.array([0, 1]), var_name=0)
    new_geop = sf_3.make_attributes_series(knots=np.arange(2), var_name='new_var')
    assert isinstance(new_geop, geopandas.GeoDataFrame)
    assert new_geop.shape[0] == sf_3.region.shape[0] * 2
    assert np.all([ni in new_geop.columns for ni in sf_3.attributes_names])
    assert 'new_var' in new_geop.columns

