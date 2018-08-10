import unittest
import numpy as np
from disarm_gears.util import binomial_to_bernoulli


# Inputs
bad_n_trials = np.random.uniform(5, 200, 20)
bad_y = np.hstack([np.random.uniform(1, bi, 1) for bi in bad_n_trials])
good_n_trials = np.ceil(bad_n_trials)
good_y = np.floor(bad_y)
good_X = np.random.normal(0, 1, good_y.size * 2).reshape(-1, 2)


class BinomialToBernoulliTests(unittest.TestCase):

    def test_inputs(self):

        self.assertRaises(AssertionError, binomial_to_bernoulli, y=bad_y, n_trials=good_n_trials)
        self.assertRaises(AssertionError, binomial_to_bernoulli, y=good_y, n_trials=bad_n_trials)
        self.assertRaises(AssertionError, binomial_to_bernoulli, y=good_y, n_trials=good_n_trials, X=good_X.flatten())

    def test_outputs(self):

        output_1 = binomial_to_bernoulli(y=good_y, n_trials=good_n_trials, X=good_X)
        self.assertEqual(len(output_1), 3)
        self.assertIsInstance(output_1[0], np.ndarray)
        self.assertIsInstance(output_1[1], np.ndarray)
        self.assertIsInstance(output_1[2], np.ndarray)
        self.assertEqual(sum(output_1[1]), sum(good_n_trials))

        output_1 = binomial_to_bernoulli(y=good_y, n_trials=good_n_trials, X=None)
        self.assertEqual(len(output_1), 3)

