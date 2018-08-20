import numpy as np
import GPy
from disarm_gears.gears.gpy_plugins import GPyRegression
from disarm_gears.validators import *


class GPyClassification(GPyRegression):

    def __init__(self):
        super(GPyClassification, self).__init__()

    def _predict_mean_gp(self, X):
        raise NotImplementedError

    def _predict_variance_gp(self, X):
        raise NotImplementedError

    def _posterior_samples_gp(self, X, n_samples=100):
        validate_2d_array(X, n_cols=self.n_dim)
        return self.base.posterior_samples_f(X, size=n_samples).T

    def _predict_variance(self, X, n_trials=None, **kwargs):
        raise NotImplementedError

    def fit(self, y, X, **kwargs):
        validate_1d_array(y)
        assert np.all(np.logical_or(y == 0, y == 1)), 'y values must be in {0, 1}.'
        validate_2d_array(X, n_rows=y.size, n_cols=None)
        self._y_train = y
        self._X_train = X
        self.n_dim = X.shape[1]
        self.base = GPy.models.GPClassification(X=X, Y=y[:, None])
        self.base.optimize()

    def predict(self, X, n_trials=None, **kwargs):
        validate_2d_array(X, n_cols=self.n_dim)
        m, _ = self.base.predict(X, full_cov=False, include_likelihood=True)
        return m.ravel()

    def posterior_samples(self, X, n_samples=100, n_trials=None, **kwargs):
        validate_2d_array(X, n_cols=self.n_dim)
        s = self.base.posterior_samples_f(X, size=n_samples).T
        return GPy.util.univariate_Gaussian.std_norm_cdf(s)
