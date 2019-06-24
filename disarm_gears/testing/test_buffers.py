import unittest
import geopandas
import numpy as np
from disarm_gears.util import voronoi_polygons, regular_polygons, disjoint_polygons


# Inputs
X = np.random.normal(0, 1, 20).reshape(-1, 2)
vrad = np.linspace(.1, 1, 10)

class BuffersTests(unittest.TestCase):

    def test_vornoi_polygons(self):

        self.assertRaises(AssertionError, voronoi_polygons, X=X.flatten())
        self.assertRaises(AssertionError, voronoi_polygons, X=X.T)

        vorpol = voronoi_polygons(X)
        self.assertIsInstance(vorpol, geopandas.GeoDataFrame)
        self.assertEqual(X.shape[0], vorpol.shape[0])

    def test_regular_polygons(self):

        self.assertRaises(AssertionError, regular_polygons, X=X.flatten(), radius=.1)
        self.assertRaises(AssertionError, regular_polygons, X=X.T, radius=.1)
        self.assertRaises(AssertionError, regular_polygons, X=X, radius=0)
        self.assertRaises(AssertionError, regular_polygons, X=X, radius=.1, n_angles=2)
        self.assertRaises(AssertionError, regular_polygons, X=X, radius=vrad[:, None], n_angles=2)
        self.assertRaises(AssertionError, regular_polygons, X=X, radius=vrad[:5], n_angles=2)

        regpol = regular_polygons(X, radius=.1, n_angles=6)
        self.assertIsInstance(regpol, geopandas.GeoDataFrame)
        self.assertEqual(X.shape[0], regpol.shape[0])

        regpol = regular_polygons(X, radius=vrad, n_angles=6)
        self.assertIsInstance(regpol, geopandas.GeoDataFrame)
        self.assertEqual(X.shape[0], regpol.shape[0])


    def test_disjoint_polygons(self):

        dispol = disjoint_polygons(X, radius=.1, n_angles=6)
        self.assertIsInstance(dispol, geopandas.GeoDataFrame)
        self.assertEqual(X.shape[0], dispol.shape[0])
