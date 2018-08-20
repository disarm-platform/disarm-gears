import unittest
import numpy as np
from disarm_gears.gears import GPyClassification

# Transformations
# Some data
n_data = 20
X = np.random.uniform(0, 2 * np.pi, 50)[:, None]
p = np.cos(X.ravel()) ** 2.
y = np.zeros_like(p)
y[p>.5] = 1

new_X = np.linspace(0, 2 * np.pi, 30)[:, None]

# Run a model
gp0 = GPyClassification()
gp0.fit(y=y, X=X)

class GPyClassificationTests(unittest.TestCase):

    def test_fit(self):

        gp = GPyClassification()
        self.assertRaises(AssertionError, gp.fit, y=y, X=X.ravel())
        self.assertRaises(AssertionError, gp.fit, y=y[:, None], X=X)
        self.assertRaises(AssertionError, gp.fit, y=p, X=X)
        self.assertTrue(hasattr(gp0, 'base'))
        self.assertTrue(hasattr(gp0, '_X_train'))
        self.assertTrue(hasattr(gp0, '_y_train'))
        self.assertTrue(hasattr(gp0, 'n_dim'))


    def test_predict_mean_gp(self):

        self.assertRaises(NotImplementedError, gp0._predict_mean_gp, X=new_X)


    def test_predict_variance_gp(self):

        self.assertRaises(NotImplementedError, gp0._predict_variance_gp, X=new_X)


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

        self.assertRaises(NotImplementedError, gp0._predict_variance, X=new_X)


    def test_posterior_samples(self):

        s1 = gp0.posterior_samples(X=new_X, n_samples=7)
        self.assertIsInstance(s1, np.ndarray)
        self.assertEqual(s1.ndim, 2)
        self.assertEqual(s1.shape[0], 7)
        self.assertEqual(s1.shape[1], new_X.shape[0])
        self.assertTrue(np.all(np.logical_and(0 <= s1, s1 <= 1.)))


    def test_log_likelihood(self):

        like = gp0._log_likelihood()
        self.assertIsInstance(like, float)

