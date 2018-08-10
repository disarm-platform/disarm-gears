import unittest
import numpy as np
from disarm_gears.util import binomial_to_bernoulli


# Inputs
bad_n = np.random.uniform(5, 200, 20)
bad_y = np.hstack([np.random.uniform(1, bi, 1) for bi in bad_n])
good_n = np.ceil(bad_n)
good_y = np.floor(bad_y)
good_X = np.random.normal(0, 1, good_y.size * 2).reshape(-1, 2)
npos_3 = np.array([2, 1, 0])
ntri_3 = np.array([2, 3, 1])

class BinomialToBernoulliTests(unittest.TestCase):

    def test_inputs(self):

        self.assertRaises(AssertionError, binomial_to_bernoulli, n_positive=bad_y, n_trials=good_n)
        self.assertRaises(AssertionError, binomial_to_bernoulli, n_positive=good_y, n_trials=bad_n)
        self.assertRaises(AssertionError, binomial_to_bernoulli, n_positive=good_y, n_trials=good_n,
                          X=good_X.flatten())

    def test_outputs(self):

        y1, w1, X1 = binomial_to_bernoulli(n_positive=good_y, n_trials=good_n, X=good_X, aggregated=False)
        self.assertIsInstance(y1, np.ndarray)
        self.assertIsInstance(w1, np.ndarray)
        self.assertIsInstance(X1, np.ndarray)
        self.assertEqual(sum(w1), sum(good_n))
        self.assertEqual(sum(y1 * w1), sum(good_y))

        output_2 = binomial_to_bernoulli(n_positive=good_y, n_trials=good_n, X=None, aggregated=False)
        self.assertEqual(len(output_2), 3)

        y3, w3, X3 = binomial_to_bernoulli(n_positive=npos_3, n_trials=ntri_3, X=None, aggregated=False)
        self.assertEqual(y3.size, sum(ntri_3))
        self.assertTrue((w3 == 1).all())
        self.assertTrue(sum(y3), sum(ntri_3))
        self.assertEqual(sum(y3 * w3), sum(npos_3))
        self.assertIsNone(X3)
