import unittest
import numpy as np
from disarm_gears.validators import *


# Inputs
An = np.random.uniform(-5, 5, 10)
In = np.arange(10).reshape(5, -1)
Pn = np.arange(1, 21).reshape(10, 2)
Anm = np.random.random(20).reshape(10, 2)
Bnm = np.random.random(30).reshape(10, 3)
Cnmk = np.random.random(30).reshape(5, 2, -1)

class ArrayValidatorsTests(unittest.TestCase):

    def test_validate_1d_array(self):
        self.assertRaises(AssertionError, validate_1d_array, x=An, size=20)
        self.assertRaises(AssertionError, validate_1d_array, x=Anm, size=None)

    def test_validate_2d_array(self):
        self.assertRaises(AssertionError, validate_2d_array, x=An, n_cols=20, n_rows=2)
        self.assertRaises(AssertionError, validate_2d_array, x=An, n_cols=10, n_rows=3)
        self.assertRaises(AssertionError, validate_2d_array, x=An, n_cols=10, n_rows=1)

    def test_validate_integer_array(self):
        self.assertRaises(AssertionError, validate_integer_array, x=An)

    def test_validate_positive_array(self):
        self.assertRaises(AssertionError, validate_positive_array, x=In)

    def test_validate_non_negative_array(self):
        self.assertRaises(AssertionError, validate_non_negative_array, x=An)
