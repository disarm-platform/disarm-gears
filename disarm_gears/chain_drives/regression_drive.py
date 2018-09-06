import numpy as np
from disarm_gears.chain_drives import SupervisedLearningCore
from disarm_gears.gears import GPyRegression#, GPyRegressionProportions
from disarm_gears.validators import *


class RegressionDrive(SupervisedLearningCore):

    def __init__(self, base_model_gen, x_norm_gen=None):
        super(RegressionDrive, self).__init__(base_model_gen=base_model_gen, x_norm_gen=x_norm_gen)

        # Check base_model is implemented
        #_bm = self.new_base_model()
        #if isinstance(_bm, GPyRegression):
        #    pass
        #elif isinstance(_bm, GPyRegressionProportions):
        #    pass
        #else:
        #    raise NotImplementedError


    def _build_yxwen(self, target, X, exposure=None, n_trials=None):
        '''Build arrays: y, X, weights, exposure and n_trials to pass to base models.'''
        #_bm = self.new_base_model()
        #if isinstance(_bm, GPyRegressionProportions):
        #    # GP Regression on transformed y: N -> [0, 1]
        #    if n_trials is None:
        #        assert np.all(np.logical_and(0 <= target, target <= 1)), 'Expecting target in [0, 1].'
        #        new_target = target
        #    else:
        #        validate_non_negative_array(target)
        #        validate_integer_array(target)
        #        assert np.all(target <= n_trials), 'Expecting n_trials >= target.'
        #        new_target = target / n_trials
        #else:
        #    new_target = target
        new_target = target

        weights = None
        exposure = exposure
        n_trials = n_trials

        return new_target, X, weights, exposure, n_trials


    def _fit_base_model(self, y, X, weights=None, exposure=None, n_trials=None, **kwargs):
        '''Train a new instance of the base_model.'''
        base_model = self.new_base_model()
        base_model.fit(y=y, X=X, weights=weights, exposure=exposure, n_trials=n_trials, **kwargs)

        return base_model


    def _predict_base_model(self, X, exposure=None, n_trials=None, phi=False, base_model=None, **kwargs):
        '''Call the prediction method of the base_model.'''
        base_model = self.base_model if base_model is None else base_model
        mu = base_model.predict(X, phi=phi, **kwargs)

        return mu


    def _posterior_samples_base_model(self, X, exposure=None, n_trials=None, n_samples=100,
                                      target=True, phi=False, base_model=None):
        '''Call the sampling method of the base_model.'''
        base_model = self.base_model if base_model is None else base_model
        samples = base_model.posterior_samples(X, n_samples=n_samples, phi=phi)

        return samples
