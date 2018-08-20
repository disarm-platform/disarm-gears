from disarm_gears.validators import *
import GPy


class GPyRegression:

    def __init__(self):
        '''
        This is a wrapper of regression models in GPy.
        See: https://github.com/SheffieldML/GPy
        '''
        pass


    def _log_likelihood(self):
        return self.base.log_likelihood()


    def _predict_mean_gp(self, X):
        validate_2d_array(X, n_cols=self.n_dim)
        m, _ = self.base.predict(X, full_cov=False, include_likelihood=False)
        return m.ravel()


    def _predict_variance_gp(self, X):
        validate_2d_array(X, n_cols=self.n_dim)
        _, v = self.base.predict(X, full_cov=False, include_likelihood=False)
        return v.ravel()


    def _predict_variance(self, X):
        validate_2d_array(X, n_cols=self.n_dim)
        _, v = self.base.predict(X, full_cov=False, include_likelihood=True)
        return v.ravel()


    def _posterior_samples_gp(self, X, n_samples=100):
        validate_2d_array(X, n_cols=self.n_dim)
        return self.base.posterior_samples_f(X, size=n_samples).T


    def fit(self, y, X, **kwargs):
        validate_1d_array(y)
        validate_2d_array(X, n_rows=y.size, n_cols=None)
        self._y_train = y
        self._X_train = X
        self.n_dim = X.shape[1]
        self.base = GPy.models.GPRegression(X=X, Y=y[:, None])
        self.base.optimize()


    def predict(self, X, *args):
        validate_2d_array(X, n_cols=self.n_dim)
        m, _ = self.base.predict(X, full_cov=False, include_likelihood=True)
        return m.ravel()


    def posterior_samples(self, X, n_samples=100):
        validate_2d_array(X, n_cols=self.n_dim)
        samples = self.base.posterior_samples(X, size=n_samples).T
        return samples

