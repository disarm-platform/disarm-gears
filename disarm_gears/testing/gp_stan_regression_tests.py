import unittest
import numpy as np
from disarm_gears.gears import GPStanRegression

# Some data
n_data = 20
X = np.random.uniform(0, 10, n_data).reshape(-1, 1)
y = X.ravel() ** 2. + np.random.normal(0, .001, n_data)
new_X = np.linspace(0, 10, 5)[:, None]

# Run a model
gp0 = GPStanRegression()
gp0.fit(y=y, X=X)

class GPStanRegressionTests(unittest.TestCase):

    def test_compile_base(self):

        self.assertTrue(gp0, 'base_train')
        self.assertTrue(gp0, 'base_prediction')


    def test_make_train_dict(self):

        d1 = gp0._make_train_dict(X=X, y=y, mu_prior=None)
        d2 = gp0._make_train_dict(X=X, y=y, mu_prior=np.zeros_like(y))
        needed_keys = ['X_data', 'n_data', 'n_dim', 'mu_data']
        self.assertIsInstance(d1, dict)
        self.assertTrue(np.all([ni in d1.keys() for ni in needed_keys]))
        self.assertIsInstance(d2, dict)
        self.assertTrue(np.all([ni in d2.keys() for ni in needed_keys]))


    def test_make_pred_dict(self):

        d1 = gp0._make_pred_dict(X=X, mu_prior=None)
        d2 = gp0._make_pred_dict(X=X, mu_prior=np.zeros_like(y))
        needed_keys = ['X_data', 'n_data', 'n_dim', 'mu_data', 'X_pred', 'n_pred', 'mu_pred']
        self.assertIsInstance(d1, dict)
        self.assertTrue(np.all([ni in d1.keys() for ni in needed_keys]))
        self.assertIsInstance(d2, dict)
        self.assertTrue(np.all([ni in d2.keys() for ni in needed_keys]))


    def fit(self):

        gp0.fit(y=y, X=X, MAP=True, prior_mean_gp=None, n_iter=200, chains=1, exposure=None)
        self.assertRaises(AssertionError, gp.fit, y=y, X=X.ravel(), MAP=True, prior_mean_gp=None,
                          n_iter=200, chains=1, exposure=None)
        self.assertRaises(AssertionError, gp.fit, y=y[:, None], X=X, MAP=True, prior_mean_gp=None,
                          n_iter=200, chains=1, exposure=None)
        self.assertRaises(AssertionError, gp.fit, y=y, X=X, MAP=3, prior_mean_gp=None,
                          n_iter=200, chains=1, exposure=None)
        self.assertTrue(hasattr(gp0, 'base'))
        self.assertTrue(hasattr(gp0, '_X_train'))
        self.assertTrue(hasattr(gp0, '_y_train'))
        self.assertTrue(hasattr(gp0, 'n_dim'))
        self.assertTrue(hasattr(gp0, 'train_dict'))
        self.assertTrue(hasattr(gp0, 'params'))
        self.assertEqual(gp0.params['cov_length'].size, gp0.n_dim)

        gp0.fit(y=y, X=X, MAP=False, prior_mean_gp=np.zeros_like(y), n_iter=200, chains=1,
                exposure=None)
        self.assertEqual(gp0.params['cov_length'].size, gp0.n_dim)


    def test_predict(self):

        mu = gp0.predict(X=new_X, MAP=True, prior_mean_gp=np.zeros(new_X.shape[0]),
                         exposure=None, n_iter=200, chains=1)
        self.assertRaises(AssertionError, gp0.predict, X=new_X, MAP=True,
                          prior_mean_gp=np.zeros(3), exposure=None, n_iter=200, chains=1)
        self.assertIsInstance(mu, np.ndarray)
        self.assertEqual(mu.ndim, 1)
        self.assertEqual(mu.size, new_X.shape[0])


    def test_posterior_samples(self):

        s1 = gp0.posterior_samples(X=new_X, n_samples=7, prior_mean_gp=np.zeros(new_X.shape[0]),
                                   exposure=None, n_iter=1000, chains=1)
        self.assertRaises(AssertionError, gp0.posterior_samples, X=new_X, MAP=True,
                          prior_mean_gp=np.zeros(3), exposure=None, n_iter=200, chains=1)
        self.assertIsInstance(s1, np.ndarray)
        self.assertEqual(s1.ndim, 2)
        self.assertEqual(s1.shape[0], 7)
        self.assertEqual(s1.shape[1], new_X.shape[0])
