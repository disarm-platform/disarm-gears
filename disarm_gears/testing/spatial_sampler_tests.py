import unittest
import numpy as np
from disarm_gears.validators import *
from disarm_gears.gears import SpatialSampler, RandomSampler, GridSearchSampler, GPyOptSampler


# A criterion function
def f1(x):
    return x[:, 0]**2. + x[:, 1]**2.

# Some domains
b_domain1 = [{'nae': 'lng', 'type': 'continuous', 'domain': (-5, 5)},
             {'name': 'lat', 'type': 'continuous', 'domain': (0, 10)}]
b_domain2 = [{'name': 'lng', 'type': 'continuous', 'domain': (-5, 5)},
             {'name': 'lat', 'type': 'mixed', 'domain': (0, 10)}]
b_domain3 = [{'name': 'lng', 'type': 'continuous', 'domain': (-5, 5)},
             {'name': 'lat', 'type': 'continuous', 'domain': (10, 0)}]
b_domain4 = [{'name': 'lng', 'type': 'continuous', 'domain': (-5, 5)},
             {'name': 'lat', 'type': 'discrete', 'domain': (0, 10)}]
g_domain1 = [{'name': 'lng', 'type': 'continuous', 'domain': (-5, 5)},
             {'name': 'lat', 'type': 'continuous', 'domain': (0, 10)}]
g_domain2 = [{'name': 'lng', 'type': 'continuous', 'domain': (-5, 5)},
             {'name': 'lat', 'type': 'discrete', 'domain': np.arange(10)}]

# A point in the domain
x1 = np.array([1, 3]).reshape(1, 2)

class SpatialSamplerTests(unittest.TestCase):

    def test_init(self):

        s1 = SpatialSampler(f=f1, domain=g_domain1, maximize=False)
        s2 = SpatialSampler(f=f1, domain=g_domain2, maximize=True)
        self.assertRaises(AssertionError, SpatialSampler, f=f1, domain=g_domain1, maximize=8)
        self.assertRaises(AssertionError, SpatialSampler, f=f1(x1), domain=b_domain3, maximize=False)
        self.assertRaises(AssertionError, SpatialSampler, f=f1(x1), domain=b_domain4, maximize=False)
        self.assertTrue(hasattr(s1, 'f'))
        self.assertTrue(hasattr(s2, 'domain'))
        self.assertTrue(hasattr(s2, 'maximize'))
        self.assertEqual(s1.f(x1), -s2.f(x1))

        # Domain validation
        self.assertRaises(AssertionError, SpatialSampler, f=f1, domain=b_domain1, maximize=False)
        self.assertRaises(AssertionError, SpatialSampler, f=f1, domain=b_domain3, maximize=False)
        self.assertRaises(NotImplementedError, SpatialSampler, f=f1, domain=b_domain2, maximize=False)

    # Already done above
    def test_validate_domain(self):
        pass

    def test_validate_set(self):

        s1 = RandomSampler(f=f1, domain=g_domain1)
        grid1 = s1._domain_grid(n_continuous=25)
        s2 = RandomSampler(f=f1, domain=g_domain2)
        self.assertRaises(AssertionError, s2._validate_set, X=grid1)

    def test_domain_grid(self):

        s2 = SpatialSampler(f=f1, domain=g_domain2, maximize=True)
        _grid = s2._domain_grid(n_continuous=5)
        self.assertRaises(AssertionError, s2._domain_grid, n_continuous=0)
        self.assertRaises(AssertionError, s2._domain_grid, n_continuous=3.4)
        self.assertIsInstance(_grid, np.ndarray)
        self.assertEqual(_grid.ndim, 2)
        self.assertEqual(_grid.shape[1], 2)
        self.assertEqual(_grid.shape[0], 5 * 10)
        self.assertEqual(np.unique(_grid[:, 0]).size, 5)
        self.assertEqual(np.unique(_grid[:, 1]).size, 10)


class RandomSamplerTests(unittest.TestCase):

    def test_choose_location(self):

        # Assert X_optim is a valid object and has the right format
        s2 = RandomSampler(f=f1, domain=g_domain2)
        s2.choose_location(X=None)
        s1 = RandomSampler(f=f1, domain=g_domain2)
        s1.choose_location(X=s2._domain_grid(n_continuous=8))
        self.assertTrue(hasattr(s2, 'X_optim'))
        self.assertIsInstance(s2.X_optim, np.ndarray)
        self.assertEqual(s2.X_optim.ndim, 2)
        self.assertEqual(s2.X_optim.shape[0], 1)
        self.assertEqual(s2.X_optim.shape[1], 2)
        self.assertEqual(s1.X_optim.ndim, 2)
        self.assertEqual(s1.X_optim.shape[0], 1)
        self.assertEqual(s1.X_optim.shape[1], 2)


class GridSearchSamplerTests(unittest.TestCase):

    def test_choose_location(self):

        # Assert X_optim is a valid object and has the right format
        s2 = GridSearchSampler(f=f1, domain=g_domain2, maximize=False)
        s2.choose_location(X=None)
        s1 = GridSearchSampler(f=f1, domain=g_domain2, maximize=False)
        s1.choose_location(X=s2._domain_grid(n_continuous=8))
        self.assertTrue(hasattr(s2, 'X_optim'))
        self.assertIsInstance(s2.X_optim, np.ndarray)
        self.assertEqual(s2.X_optim.ndim, 2)
        self.assertEqual(s2.X_optim.shape[0], 1)
        self.assertEqual(s2.X_optim.shape[1], 2)
        self.assertEqual(s1.X_optim.ndim, 2)
        self.assertEqual(s1.X_optim.shape[0], 1)
        self.assertEqual(s1.X_optim.shape[1], 2)

        # Assert optimal is being found
        X_grid = np.array([[-1, 2], [0, 0], [2, 0], [-4, 4]])
        s1.choose_location(X=X_grid)
        self.assertEqual(s1.X_optim[0, 0], 0)
        self.assertEqual(s1.X_optim[0, 1], 0)


class GPyOptSamplerTests(unittest.TestCase):

    def test_choose_location(self):

        # Assert X_optim is a valid object and has the right format
        s1 = GPyOptSampler(f=f1, domain=g_domain1, maximize=False)
        s1.choose_location()
        self.assertTrue(hasattr(s1, 'X_optim'))
        self.assertIsInstance(s1.X_optim, np.ndarray)
        self.assertEqual(s1.X_optim.ndim, 2)
        self.assertEqual(s1.X_optim.shape[0], 1)
        self.assertEqual(s1.X_optim.shape[1], 2)

        # Assert optimal is being found

