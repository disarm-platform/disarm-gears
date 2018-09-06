import pystan
from disarm_gears.gears.stan_plugins import stan_compilers
from disarm_gears.validators import *


class GPStanRegression:

    def __init__(self, stan_models=None):
        '''
        This is a wrapper of a model implemented in pystan.
        See: https://mc-stan.org
        '''

        if stan_models is not None:
            assert len(stan_models) == 2, 'Expecting list or tuple of length 2.'
            assert isinstance(stan_models[0], pystan.model.StanModel)
            assert isinstance(stan_models[1], pystan.model.StanModel)
            self.base_train, self.base_prediction = stan_models
            self.base_train = stan_models[0]
            self.base_prediction = stan_models[1]
        else:
            stan_models = self._compile_models()


    def _compile_models(self):
        return stan_compilers.gp_regression_compiler()


    def _make_train_dict(self, X, y, mu_prior, **kwargs):

        return {'X_data': X, 'y_data': y, 'n_data': y.size, 'n_dim': X.shape[1],
                'mu_data': mu_prior}


    def _make_pred_dict(self, X, mu_prior, **kwargs):

        assert hasattr(self, 'train_dict')
        new_dict = self.train_dict.copy()
        new_dict.update(self.params)
        new_dict.update({'X_pred': X, 'n_pred': X.shape[0], 'mu_pred': mu_prior})

        return new_dict


    def fit(self, y, X, MAP=True, prior_mean_gp=None, n_iter=1000, chains=1, exposure=None,
            n_trials=None, **kwargs):

        #validate_1d_array(y)
        #validate_2d_array(X, n_rows=y.size, n_cols=None)
        assert isinstance(MAP, bool)

        self._y_train = y
        self._X_train = X
        self.n_dim = X.shape[1]

        # Set prior mean of the GP
        if prior_mean_gp is not None:
            validate_1d_array(prior_mean_gp, size=y.size)
        else:
            prior_mean_gp = np.zeros_like(y)

        # Data dictionary to pass to Stan
        if exposure is None:
            exposure = np.ones_like(y)
        if n_trials is None:
            n_trials = np.ones_like(y)

        self.train_dict = self._make_train_dict(X=X, y=y, mu_prior=prior_mean_gp,
                                                exposure=exposure, n_trials=n_trials)

        # Maximum a Posteriori or Sampling
        if MAP:
            base = self.base_train.optimizing(data=self.train_dict)
            self.params = {par: val for par, val in zip(base.keys(), base.values())}
        else:
            base = self.base_train.sampling(data=self.train_dict, iter=n_iter, chains=chains)
            base_samples = base.extract()
            self.params = {par: smp.mean(0) for par, smp in zip(base_samples.keys(),
                                                                base_samples.values())}

        # cov_length is declared as vector to use ARD
        if self.n_dim == 1:
            self.params['cov_length'] = self.params['cov_length'].reshape(self.n_dim)


    def predict(self, X, MAP=False, prior_mean_gp=None, exposure=None, n_trials=None,
                phi=False, n_iter=1000, chains=1, **kwargs):
        validate_2d_array(X, n_cols=self.n_dim)
        assert isinstance(MAP, bool), 'MAP must be boolean.'

        # Set prior mean of the GP
        if prior_mean_gp is None:
            prior_mean_gp = np.zeros(X.shape[0])
        else:
            validate_1d_array(prior_mean_gp, size=X.shape[0])

        # exposure and n_trials
        if exposure is None:
            exposure = np.ones(X.shape[0])
        if n_trials is None:
            n_trials = np.ones(X.shape[0])

        if not MAP:
            s = self.posterior_samples(X=X, n_samples=int(n_iter*.5), prior_mean_gp=prior_mean_gp,
                                       exposure=exposure, n_trials=n_trials, phi=phi,
                                       n_iter=n_iter, chains=chains)
            m = s.mean(0)
        else:
            # Data dictionary to pass to Stan
            pred_dict = self._make_pred_dict(X=X, mu_prior=prior_mean_gp, exposure=exposure,
                                             n_trials=n_trials)
            base = self.base_prediction.optimizing(data=pred_dict)
            m = base.get('phi_pred') if phi else base.get('y_pred')

        return m


    def posterior_samples(self, X, n_samples=100, prior_mean_gp=None, exposure=None,
                          n_trials=None, phi=True, n_iter=1000, chains=1, **kwargs):
        validate_2d_array(X, n_cols=self.n_dim)
        assert isinstance(n_samples, int), 'n_samples is expected to be an integer.'
        assert n_samples > 0, 'n_samples is expected to be a positive number.'

        #TODO: This is an inefficient temporal solution
        if n_iter < n_samples:
            n_iter += n_samples

        # Set prior mean of the GP
        if prior_mean_gp is None:
            prior_mean_gp = np.zeros(X.shape[0])
        else:
            validate_1d_array(prior_mean_gp, size=X.shape[0])

        # Data dictionary to pass to Stan
        if exposure is None:
            exposure = np.ones(X.shape[0])
        if n_trials is None:
            n_trials = np.ones(X.shape[0])
        pred_dict = self._make_pred_dict(X=X, mu_prior=prior_mean_gp, exposure=exposure,
                                         n_trials=n_trials)
        base = self.base_prediction.sampling(data=pred_dict, iter=n_iter, chains=chains)
        s = base.extract()['phi_pred'] if phi else base.extract()['y_pred']

        return s[:n_samples, ]

