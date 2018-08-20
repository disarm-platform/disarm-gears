import numpy as np
import GPy
from disarm_gears.gears.gpy_plugins import GPyRegression
from disarm_gears.validators import *


class GPyRegressionProportions(GPyRegression):

    def __init__(self):
        super(GPyRegressionProportions, self).__init__()

    def _transform_f(self, f):
        return 1. / (1 + np.exp(-f))

    def _transform_y(self, y):
        y = np.clip(y, 1e-8, 1 - 1e-8)
        return np.log(y / (1-y))

    def _predict_mean_gp(self, X):
        validate_2d_array(X, n_cols=self.n_dim)
        m, _ = self.base.predict(X, full_cov=False, include_likelihood=False)
        return m.ravel()

    def _predict_variance_gp(self, X):
        validate_2d_array(X, n_cols=self.n_dim)
        _, v = self.base.predict(X, full_cov=False, include_likelihood=False)
        return v.ravel()

    def _posterior_samples_gp(self, X, n_samples=100):
        validate_2d_array(X, n_cols=self.n_dim)
        return self.base.posterior_samples_f(X, size=n_samples).T

    def _predict_variance(self, X, n_trials=None, **kwargs):
        validate_2d_array(X, n_cols=self.n_dim)
        s = self.posterior_samples(X, n_trials=n_trials)
        return s.var(0)

    def fit(self, y, X, **kwargs):
        validate_1d_array(y)
        validate_2d_array(X, n_rows=y.size, n_cols=None)
        self._y_train = y
        self._X_train = X
        self.n_dim = X.shape[1]
        Y_transformed = self._transform_y(y)[:, None]
        self.base = GPy.models.GPRegression(X=X, Y=Y_transformed)
        self.base.optimize()

    def predict(self, X, n_trials=None, **kwargs):
        validate_2d_array(X, n_cols=self.n_dim)
        s = self.posterior_samples(X, n_trials=n_trials).mean(0)
        return s

    def posterior_samples(self, X, n_samples=100, n_trials=None, **kwargs):
        validate_2d_array(X, n_cols=self.n_dim)
        s = self._posterior_samples_gp(X ,n_samples=n_samples)
        s = self._transform_f(s)
        if n_trials is not None:
            validate_1d_array(n_trials, size=X.shape[0])
            s *= n_trials
        return s
