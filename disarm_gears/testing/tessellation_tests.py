import unittest
import numpy as np
import pandas as pd
import geopandas as geop
from disarm_gears.frames import Tessellation


# Inputs
b_points_1 = np.random.uniform(0, 1, 10)
b_points_2 = np.random.uniform(0, 1, 30).reshape(10, 3)
g_points = np.random.uniform(0, 1, 20).reshape(10, -1)
b_attrib = np.random.random(25)
g_attrib_1 = np.random.random(10)
g_attrib_2 = np.random.random(40).reshape(10, -1)
g_attrib_3 = pd.DataFrame({li:ci for li,ci in zip(['a', 'b', 'c', 'd'], g_attrib_2.T)})
n_points = g_points.shape[0]
X = np.vstack([g_points.copy()[5:], np.array([10, 10])])

class TessellationTests(unittest.TestCase):

    def test_inputs(self):

        # Check bad inputs
        self.assertRaises(AssertionError, Tessellation, points=b_points_1)
        self.assertRaises(AssertionError, Tessellation, points=b_points_2)
        self.assertRaises(AssertionError, Tessellation, points=g_points, attributes=b_attrib)
        self.assertRaises(AssertionError, Tessellation, points=g_points, attributes=None, crs=0)

    def test_outputs(self):

        # Check output types
        sf_1 = Tessellation(points=g_points, attributes=None, crs=None)
        sf_2 = Tessellation(points=g_points, attributes=g_attrib_1, crs=None)
        sf_3 = Tessellation(points=g_points, attributes=g_attrib_2, crs=None)
        sf_4 = Tessellation(points=g_points, attributes=g_attrib_3, crs=None)

        # Check sf.region is geop.GeoDataFrame
        self.assertIsInstance(sf_2.centroids, np.ndarray)
        self.assertEqual(sf_4.centroids.shape[0], n_points)
        self.assertIsInstance(sf_1.region, geop.GeoDataFrame)
        self.assertIsInstance(sf_3.region, geop.GeoDataFrame)

        # Check sf.region shape
        self.assertEqual(sf_1.region.ndim, 2)
        self.assertEqual(sf_1.region.shape[0], n_points)
        self.assertEqual(sf_3.region.shape[0], n_points)
        self.assertEqual(sf_1.region.shape[1], 1)
        self.assertEqual(sf_2.region.shape[1], 2)
        self.assertEqual(sf_4.region.shape[1], 5)

        # Check sf.region.columns
        self.assertTrue('geometry' in sf_3.region.columns)
        self.assertTrue('geometry' in sf_4.region.columns)

        # Check attribute names
        self.assertTrue(np.array('var_%s' %i in sf_3.region.columns for i in range(4)).all())
        self.assertTrue(np.array(v in sf_3.region.columns for v in ['a', 'b', 'c', 'd']).all())

        # Check box type
        self.assertIsInstance(sf_2.box, pd.DataFrame)
        self.assertEqual(sf_3.box.ndim, 2)
        self.assertEqual(sf_1.box.shape[0], 2)
        self.assertEqual(sf_4.box.shape[1], 2)

    def test_locate(self):

        sf_1 = Tessellation(points=g_points, attributes=None, crs=None)
        ix = sf_1.locate(X=X)
        self.assertIsInstance(ix, np.ndarray)
        self.assertTrue(ix.ndim, 1)
        self.assertEqual(ix.size, X.shape[0])
        self.assertEqual(ix[-1], -1)
        self.assertTrue((ix[:-1] - np.arange(5, 10) == 0).all())
