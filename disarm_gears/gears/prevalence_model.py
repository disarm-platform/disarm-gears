import numpy as np
from disarm_gears.gears import GaussianProcess

# TEMPORAL SOLUTION: for the time being this is good enough!!
class PrevalenceModel(GaussianProcess):

    def __init__(self):
        super(GaussianProcess, self).__init__()


    def _activation(self, x):
        return 1./(1. + np.exp(-np.clip(x, -250, 250)))


    def predict(self, X):
        return self._activation(self.predict_mean_gp(X))


    def predict_variance(self, X, diagonal=False):

        noise_var = np.exp(self._raw_noise_var)

        if diagonal:
            Cov = self.predict_variance_gp(X, diagonal=diagonal) + noise_var
        else:
            Cov = self.predict_variance_gp(X, diagonal=diagonal) + noise_var * np.eye(X.shape[0])

        return Cov


    def posterior_samples(self, X, n_samples=100):

        mean = self.predict_mean_gp(X)
        Cov = self.predict_variance(X, diagonal=False)
        samples = np.random.multivariate_normal(mean, Cov, size=n_samples)

        return self._activation(samples)
