import numpy as np
from .supervised_learning_core import SupervisedLearningCore
from ..util import binomial_to_bernoulli
from ..gears import PrevalenceModel


class ClassificationPipeline(SupervisedLearningCore):

    def __init__(self, base_model_gen, x_norm_gen=None):
        super(ClassificationPipeline, self).__init__(base_model_gen=base_model_gen, x_norm_gen=x_norm_gen)

        # Check base_model is implemented
        base_model = self.new_base_model()
        if isinstance(base_model, PrevalenceModel):
            pass
        else:
            raise NotImplementedError


    def _build_yxwe(self, target, X, n_trials=None, exposure=None):
        '''Build arrays: y, X, weights and exposure to pass to base models.'''
        new_target, weights, new_X = binomial_to_bernoulli(n_positive=target, n_trials=n_trials, X=X,
                                                           aggregated=True)
        exposure = None

        return new_target, new_X, weights, exposure

    def _fit_base_model(self, y, X, weights, exposure=None):
        '''Train a new instance of the base_model.'''
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
        else:
            raise NotImplementedError

        return base_model


    def _predict_base_model(self, X, exposure=None, base_model=None):
        '''Call the prediction method of the base_model.'''
        base_model = self.base_model if base_model is None else base_model
        if isinstance(base_model, PrevalenceModel):
            mu = base_model.predict(X)
        else:
            raise NotImplementedError

        return mu


    def _posterior_samples_base_model(self, X, exposure=None, n_samples=100, base_model=None):
        '''Call the sampling method of the base_model.'''
        base_model = self.base_model if base_model is None else base_model
        if isinstance(base_model, PrevalenceModel):
            samples = base_model.posterior_samples(X, n_samples=n_samples)
        else:
            raise NotImplementedError

        return samples

