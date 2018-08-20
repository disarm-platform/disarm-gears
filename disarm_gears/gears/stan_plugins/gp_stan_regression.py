import pystan
from pathlib import Path
from disarm_gears.validators import *


class GPStanRegression:

    def __init__(self):
        '''
        This is a wrapper of a model implemented in pystan.
        See: https://mc-stan.org
        '''
        self._compile_base()


    def _compile_base(self):
        train_script = 'disarm_gears/gears/stan_plugins/stan_scripts/gp_gaussian_fit_hyper.stan'
        prediction_script = 'disarm_gears/gears/stan_plugins/stan_scripts/gp_gaussian_predict.stan'
        assert Path(train_script).exists(), '%s not found.' %train_script
        assert Path(prediction_script).exists(), '%s not found.' %prediction_script
        self.base_train = pystan.StanModel(file=train_script)
        self.base_prediction = pystan.StanModel(file=prediction_script)


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
            **kwargs):

        validate_1d_array(y)
        validate_2d_array(X, n_rows=y.size, n_cols=None)
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
        self.train_dict = self._make_train_dict(X=X, y=y, mu_prior=prior_mean_gp, exposure=exposure)

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


    def predict(self, X, MAP=False, prior_mean_gp=None, exposure=None, n_iter=1000, chains=1, **kwargs):
        validate_2d_array(X, n_cols=self.n_dim)
        assert isinstance(MAP, bool), 'MAP must be boolean.'

        # Set prior mean of the GP
        if prior_mean_gp is None:
            prior_mean_gp = np.zeros(X.shape[0])
        else:
            validate_1d_array(prior_mean_gp, size=X.shape[0])

        if not MAP:
            s = self.posterior_samples(X=X, n_samples=int(n_iter*.5), prior_mean_gp=prior_mean_gp,
                                       n_iter=n_iter, chains=chains)
            m = s.mean(0)
        else:
            # Data dictionary to pass to Stan
            pred_dict = self._make_pred_dict(X=X, mu_prior=prior_mean_gp, exposure=exposure)
            base = self.base_prediction.optimizing(data=pred_dict)
            m = base.get('y_pred')

        return m


    def posterior_samples(self, X, n_samples=100, prior_mean_gp=None, exposure=None, n_iter=1000, chains=1,
                          **kwargs):
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
        pred_dict = self._make_pred_dict(X=X, mu_prior=prior_mean_gp, new_exposure=exposure)
        base = self.base_prediction.sampling(data=pred_dict, iter=n_iter, chains=chains)
        s = base.extract()['y_pred']

        return s[:n_samples, ]
