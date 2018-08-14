import numpy as np
from .chain_drive import ChainDrive
from ..util import binomial_to_bernoulli
from ..gears import PrevalenceModel


class ClassificationChain(ChainDrive):

    def __init__(self, base_model_gen, x_norm_gen=None):
        super(ClassificationChain, self).__init__(base_model_gen=base_model_gen, x_norm_gen=x_norm_gen)


    def _build_yxwe(self, target, X, n_trials=None, exposure=None):
        '''Build arrays: y, X, weights and exposure to pass to base models.'''

        new_target, weights, new_X = binomial_to_bernoulli(n_positive=target, n_trials=n_trials, X=X,
                                                           aggregated=True)
        exposure = None

        return new_target, new_X, weights, exposure

    def _fit_base_model(self, y, X, weights, exposure=None):

        base_model = self.new_base_model()

        if isinstance(base_model, PrevalenceModel):
            y[y == 0] -= 1.

            j = 0
            if self.spatial:
                slice_s = slice(0, 2)
                j += 2
            else:
                slice_s = None
            if self.temporal:
                slice_t = slice(j, j+1)
                j += 1
            else:
                slice_t = None
            if self.n_features > 0:
                slice_f = slice(j, j + self.n_features)
            else:
                slice_f = None

        base_model.fit(y=y, X=X, weights=weights, slice_s=slice_s, slice_t=slice_t, slice_f=slice_f)

        return base_model


    def _predict_base_model(self, X, exposure=None):

        mu = self.base_model.predict(X)
        return mu


    def _posterior_samples_base_model(self, X, exposure=None, n_samples=100):

        return self.base_model.posterior_samples(X, n_samples=n_samples)

