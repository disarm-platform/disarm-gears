'''
import unittest
import numpy as np
from disarm_gears.gears import GPyRegressionProportions

# Transformations
def tf(f):
    return 1. / (1. + np.exp(-f))

def ty(y):
    return np.log(y / (1 - y))

# Some data
n_data = 20
X = np.random.uniform(0, 10, n_data).reshape(-1, 1)
f = X.ravel() ** 2. + np.random.normal(0, .001, n_data)
y = tf(f)
new_X = np.linspace(0, 10, 5)[:, None]

# Run a model
gp0 = GPyRegressionProportions()
gp0.fit(y=y, X=X)

class GPyRegressionProportionsTests(unittest.TestCase):

    def test_fit(self):

        gp = GPyRegressionProportions()
        self.assertRaises(AssertionError, gp.fit, y=y, X=X.ravel())
        self.assertRaises(AssertionError, gp.fit, y=y[:, None], X=X)
        self.assertTrue(hasattr(gp0, 'base'))
        self.assertTrue(hasattr(gp0, '_X_train'))
        self.assertTrue(hasattr(gp0, '_y_train'))
        self.assertTrue(hasattr(gp0, 'n_dim'))


    def test_predict_mean_gp(self):

        mu = gp0._predict_mean_gp(X=new_X)
        self.assertIsInstance(mu, np.ndarray)
        self.assertEqual(mu.ndim, 1)
        self.assertEqual(mu.size, new_X.shape[0])


    def test_predict_variance_gp(self):

        v1 = gp0._predict_variance_gp(X=new_X)
        self.assertIsInstance(v1, np.ndarray)
        self.assertEqual(v1.ndim, 1)
        self.assertEqual(v1.size, new_X.shape[0])


    def test_posterior_samples_gp(self):

        s1 = gp0._posterior_samples_gp(X=new_X, n_samples=7)
        self.assertIsInstance(s1, np.ndarray)
        self.assertEqual(s1.ndim, 2)
        self.assertEqual(s1.shape[0], 7)
        self.assertEqual(s1.shape[1], new_X.shape[0])


    def test_predict(self):

        mu = gp0.predict(X=new_X)
        self.assertIsInstance(mu, np.ndarray)
        self.assertEqual(mu.ndim, 1)
        self.assertEqual(mu.size, new_X.shape[0])
        self.assertTrue(np.all(np.logical_and(0 <= mu, mu <= 1.)))


    def test_predict_variance(self):

        v1 = gp0._predict_variance(X=new_X)
        self.assertIsInstance(v1, np.ndarray)
        self.assertEqual(v1.ndim, 1)
        self.assertEqual(v1.size, new_X.shape[0])


    def test_posterior_samples(self):

        s1 = gp0.posterior_samples(X=new_X, n_samples=7)
        self.assertIsInstance(s1, np.ndarray)
        self.assertEqual(s1.ndim, 2)
        self.assertEqual(s1.shape[0], 7)
        self.assertEqual(s1.shape[1], new_X.shape[0])


    def test_log_likelihood(self):

        like = gp0._log_likelihood()
        self.assertIsInstance(like, float)
'''
