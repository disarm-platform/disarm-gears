import numpy as np
import scipy
from ..validators import *


class GaussianProcess:

    def __init__(self):
        '''General class for Gaussian process models.'''
        pass


    def fit(self, y, X, weights=None, exposure=None, slice_s=None, slice_t=None, slice_f=None):
        '''Fit a Gaussian process to y, X.

        :param y: Response variable.
                  Numpy array, shape [n_data, ].
        :param X: GP inputs.
                  Numpy array, shape [n_data, ] or [n_data, n_dim].
        :param weights: Observation weights (optional).
                        Numpy array, shape [n_data, ].
        :param exposure: Exposure associated to the response variable (optional).
                         Numpy array, shape [n_data, ].
        :param slice_s: Column-slices that correspond to spatial coordinates (optional).
                        Slice.
        :param slice_t: Column-slices that correspond to a time reference (optional).
                        Slice.
        :param slice_f: Column-slices that correspond to features that are not space or time (optional).
                        Slice.
        '''
        # Check y, X dims
        validate_1d_array(y)
        if X.ndim == 1:
            X = np.array(X).reshape(-1, 1)
        validate_2d_array(X, n_rows=y.size)

        assert len([si for si in [slice_s, slice_t, slice_f] if si is not None]) > 0
        self._store_raw_inputs_dims(y=y, X=X, slice_s=slice_s, slice_t=slice_t, slice_f=slice_f)

        # Weights
        if weights is not None:
            validate_1d_array(weights, size=y.size)
            validate_positive_array(weights)
            validate_integer_array(weights)
            weights = weights.astype(int)
            y = np.hstack([np.repeat(yi, wi) for yi,wi in zip(y, weights)])
            self._Ex = self._expansion_matrix(weights=weights)
        self._weights = weights
        self._X_train = X
        self._y_train = y

        self._initialize_parameters(n_dim=X.shape[1], slice_s=slice_s, slice_t=slice_t, slice_f=slice_f)
        self._distance2_X_train()

        #TODO fit and store: _kernel_train, L


    def predict_mean_gp(self, X):

        if X.ndim == 1:
            X = np.array(X).reshape(-1, 1)
        validate_2d_array(X, n_cols=self.n_dim)

        K_nm = self._kernel_prediction_cross(X)
        noise_var = np.exp(self._raw_noise_var)
        K = self._kernel_train()

        if self._weights is None:
            K_noise =  K + noise_var * np.eye(self.n_data)
            L = scipy.linalg.cholesky(K_noise, lower=True)
            mu = np.dot(K_nm.T, scipy.linalg.solve(L.T, scipy.linalg.solve(L, self._y_train)))
        else:
            KET = np.dot(self._Ex, K_nm).T
            L1 = scipy.linalg.cholesky(K + np.eye(self.n_data) * 1e-8, lower=True)
            L_inv = scipy.linalg.solve(L1, np.eye(self.n_data))
            K_inv = scipy.linalg.solve(L1.T, L_inv) * noise_var
            Q = scipy.linalg.cholesky(K_inv + np.dot(self._Ex.T, self._Ex), lower=True)
            a = scipy.linalg.solve(Q.T, scipy.linalg.solve(Q, np.dot(self._Ex.T, self._y_train)))
            mu = np.dot(KET, self._y_train) / noise_var - np.dot(KET, np.dot(self._Ex, a)) / noise_var

        return mu


    def predict_variance_gp(self, X, diagonal=True):


        if X.ndim == 1:
            X = np.array(X).reshape(-1, 1)
        validate_2d_array(X, n_cols=self.n_dim)

        K_nm = self._kernel_prediction_cross(X)
        K_mm = self._kernel_prediction_cov(X)
        m = K_mm.shape[0]
        noise_var = np.exp(self._raw_noise_var)
        K = self._kernel_train()

        if self._weights is None:
            K_noise = K + noise_var * np.eye(self.n_data)
            L = scipy.linalg.cholesky(K_noise)
            Cov = K_mm - np.dot(K_nm.T, scipy.linalg.solve(L, scipy.linalg.solve(L.T, K_nm)))
        else:
            KET = np.dot(self._Ex, K_nm).T
            L1 = scipy.linalg.cholesky(K + np.eye(self.n_data) * 1e-8, lower=True)
            L_inv = scipy.linalg.solve(L1, np.eye(self.n_data))
            K_inv = scipy.linalg.solve(L1.T, L_inv) * noise_var
            Q = scipy.linalg.cholesky(K_inv + np.dot(self._Ex.T, self._Ex), lower=True)
            a = scipy.linalg.solve(Q.T, scipy.linalg.solve(Q, np.dot(self._Ex.T, KET.T)))
            Cov = K_mm - np.dot(KET, KET.T) / noise_var + np.dot(KET, np.dot(self._Ex, a)) / noise_var

        if diagonal:
            v = np.diag(Cov) + noise_var
        else:
            v = Cov + np.eye(m) * noise_var

        return v


    def posterior_samples_gp(self, X, n_samples=100):

        mean = self.predict_mean_gp(X)
        Cov = self.predict_variance_gp(X, diagonal=False)
        samples = np.random.multivariate_normal(mean, Cov, size=n_samples)

        return samples


    def predict(self, X):
        return self.predict_mean_gp(X)


    def predict_variance(self, X, diagonal=False):

        noise_var = np.exp(self._raw_noise_var)

        if diagonal:
            Cov = self.predict_variance_gp(X, diagonal=diagonal) + noise_var
        else:
            Cov = self.predict_variance_gp(X, diagonal=diagonal) + noise_var * np.eye(X.shape[0])

        return Cov

    def posterior_samples(self, X, n_samples=100):

        mean = self.predict(X)
        Cov = self.predict_variance(X, diagonal=False)
        samples = np.random.multivariate_normal(mean, Cov, size=n_samples)

        return samples


    def log_likelihood(self):

        noise_var = np.exp(self._raw_noise_var)
        K = self._kernel_train()
        K_noise = K + noise_var * np.eye(self.n_data)
        D = scipy.linalg.det(K_noise)
        L = scipy.linalg.cholesky(K_noise)
        yLLiy = np.dot(self._y_train, scipy.linalg.solve(L, scipy.linalg.solve(L.T, self._y_train)))

        # Weights
        if self._weights is not None:
            pass
            #TODO

        return - .5 * self.n_data * np.log(2 * np.pi) - .5 * np.log(D) - .5 * yLLiy


    def _store_raw_inputs_dims(self, y, X, slice_s, slice_t, slice_f):
        '''Store the inputs dimensions.'''

        # Store dimensions
        self.n_data = y.size
        self.n_dim = X.shape[1]
        self.spatial = False if slice_s is None else True
        self.temporal = False if slice_t is None else True
        self.n_features = 0 if slice_f is None else len(list(range(*slice_f.indices(self.n_dim))))

        # Store slices
        if self.spatial:
            self.slice_s = slice_s
        if self.temporal:
            self.slice_t = slice_t
        if self.n_features > 0:
            self.slice_f = slice_f


    def _initialize_parameters(self, n_dim, slice_s, slice_t, slice_f):
        '''Initialize parameters in the Real space.'''

        k_var_dim = 0
        k_len_dim = 0
        vi = 0
        li = 0

        # Parameters for the spatial dimension
        if self.spatial:
            k_var_dim += 1
            k_len_dim += 2
            self._slice_var_s = slice(vi, vi + 1)
            self._slice_len_s = slice(li, li + 2)
            vi += 1
            li += 2

        # Parameters for the temporal dimension
        if self.temporal:
            k_var_dim += 1
            k_len_dim += 1
            self._slice_var_t = slice(vi, vi + 1)
            self._slice_len_t = slice(li, li + 1)
            vi += 1
            li += 1

        # Parameters for the features dimensions
        if self.n_features > 0:
            k_var_dim += 1
            k_len_dim += self.n_features
            self._slice_var_f = slice(vi, vi + 1)
            self._slice_len_f = slice(li, li + self.n_features)
            vi += 1
            li += self.n_features

        self._raw_k_var = np.repeat(.7, k_var_dim)
        self._raw_k_len = np.repeat(0, k_len_dim)

        # Noise parameter
        self._raw_noise_var = -9.


    def _distance2_X_train(self):
        '''Compute squared distances between columns of X in training set.'''
        if self.spatial:
            self._dist2_s =\
                (self._X_train[:, self.slice_s][:, None] - self._X_train[:, self.slice_s][None, :])**2.
        if self.temporal:
            self._dist2_t =\
                (self._X_train[:, self.slice_t][:, None] - self._X_train[:, self.slice_t][None, :])**2.
        if self.n_features > 0:
            self._dist2_f =\
                (self._X_train[:, self.slice_f][:, None] - self._X_train[:, self.slice_f][None, :])**2.


    def _kernel_train(self):
        '''Compute K(X_train, X_train) for the training set.'''

        k_var = np.exp(self._raw_k_var)
        k_2len2 = 2 * np.exp(self._raw_k_len)**2.

        K = 0
        if self.spatial:
            K += k_var[self._slice_var_s] * np.exp(- (self._dist2_s / k_2len2[self._slice_len_s]).sum(2))
        if self.temporal:
            K += k_var[self._slice_var_t] * np.exp(- (self._dist2_t / k_2len2[self._slice_len_t]).sum(2))
        if self.n_features > 0:
            K += k_var[self._slice_var_f] * np.exp(- (self._dist2_f / k_2len2[self._slice_len_f]).sum(2))

        return K


    def _kernel_prediction_cross(self, X):
        '''Compute K(X_train, X_new) between the training set and an new set of input X.'''

        k_var = np.exp(self._raw_k_var)
        k_2len2 = 2 * np.exp(self._raw_k_len)**2.

        K = 0
        if self.spatial:
            dist2_s = (self._X_train[:, self.slice_s][:, None] - X[:, self.slice_s][None, :])**2.
            K += k_var[self._slice_var_s] * np.exp(- (dist2_s / k_2len2[self._slice_len_s]).sum(2))
        if self.temporal:
            dist2_t = (self._X_train[:, self.slice_t][:, None] - X[:, self.slice_t][None, :])**2.
            K += k_var[self._slice_var_t] * np.exp(- (dist2_t / k_2len2[self._slice_len_t]).sum(2))
        if self.n_features > 0:
            dist2_f = (self._X_train[:, self.slice_f][:, None] - X[:, self.slice_f][None, :])**2.
            K += k_var[self._slice_var_f] * np.exp(- (dist2_f / k_2len2[self._slice_len_f]).sum(2))

        return K


    def _kernel_prediction_cov(self, X):
        '''Compute K(X_train, X_new) between the training set and an new set of input X.'''

        k_var = np.exp(self._raw_k_var)
        k_2len2 = 2 * np.exp(self._raw_k_len)**2.

        K = 0
        if self.spatial:
            dist2_s = (X[:, self.slice_s][:, None] - X[:, self.slice_s][None, :])**2.
            K += k_var[self._slice_var_s] * np.exp(- (dist2_s / k_2len2[self._slice_len_s]).sum(2))
        if self.temporal:
            dist2_t = (X[:, self.slice_t][:, None] - X[:, self.slice_t][None, :])**2.
            K += k_var[self._slice_var_t] * np.exp(- (dist2_t / k_2len2[self._slice_len_t]).sum(2))
        if self.n_features > 0:
            dist2_f = (X[:, self.slice_f][:, None] - X[:, self.slice_f][None, :])**2.
            K += k_var[self._slice_var_f] * np.exp(- (dist2_f / k_2len2[self._slice_len_f]).sum(2))

        return K


    def _expansion_matrix(self, weights):
        '''Returns a matrix to expand a covariance matrix  based on a set of weights.'''
        Ex = np.zeros([weights.sum(), weights.size])
        i = 0
        for j, wj in enumerate(weights):
            Ex[slice(i, i+wj), j] = 1
            i += wj

        return Ex

