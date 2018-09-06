import unittest
import numpy as np
from disarm_gears.util import trend_1st_order, trend_2nd_order, trend_3rd_order

X = np.arange(6).reshape(3, 2)

class TessellationTests(unittest.TestCase):

    def test_trend_1st_order(self):
        Z = trend_1st_order(X)
        self.assertIsInstance(Z, np.ndarray)
        self.assertEqual(Z.shape[0], 3)
        self.assertEqual(Z.shape[1], 3)
        self.assertEqual(Z[2, 2], 20)

    def test_trend_2nd_order(self):
        Z = trend_2nd_order(X)
        self.assertIsInstance(Z, np.ndarray)
        self.assertEqual(Z.shape[0], 3)
        self.assertEqual(Z.shape[1], 5)
        self.assertEqual(Z[1, 0], 4)
        self.assertEqual(Z[2, 1], 25)
        self.assertEqual(Z[0, 2], 0)
        self.assertEqual(Z[1, 3], 3)
        self.assertEqual(Z[2, 4], 20)

    def test_trend_3rd_order(self):
        Z = trend_3rd_order(X)
        self.assertIsInstance(Z, np.ndarray)
        self.assertEqual(Z.shape[0], 3)
        self.assertEqual(Z.shape[1], 9)
        self.assertEqual(Z[1, 0], 8)
        self.assertEqual(Z[2, 1], 125)
        self.assertEqual(Z[1, 2], 4)
        self.assertEqual(Z[2, 3], 25)
        self.assertEqual(Z[0, 4], 0)
        self.assertEqual(Z[1, 5], 3)
        self.assertEqual(Z[1, 6], 12)
        self.assertEqual(Z[1, 7], 18)
        self.assertEqual(Z[2, 8], 20)

