import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from disarm_gears.validators import *


class SupervisedLearningCore(object):

    def __init__(self, base_model_gen, x_norm_gen=None):
        '''
        Core pipeline for supervised models.

        :param base_model_gen: A callable object that instantiates a machine learning model.
        :param x_norm_gen: A callable object that instantiates a preprocessing method.
                           One of StandardScaler or Normalizer from scikit-learn.
        '''
        assert callable(base_model_gen), 'Object is not callable.'
        self.new_base_model = base_model_gen

        # Define x normalizer
        if x_norm_gen is not None:
            assert callable(x_norm_gen), 'Object is not callable.'
            assert isinstance(x_norm_gen(), StandardScaler) or isinstance(x_norm_gen(), Normalizer)
            self.x_norm = True
            self.new_x_norm_rule = x_norm_gen
        else:
            self.x_norm = False


    def _store_raw_inputs_dims(self, target, x_coords, x_time, x_features):
        '''Store the dimensions of the input objects.'''
        self.n_data = target.size
        self.spatial = False if x_coords is None else True
        self.temporal = False if x_time is None else True
        self.n_features = 0 if x_features is None else x_features.shape[1]


    def _validate_train_inputs_dims(self, target, x_coords, x_time, x_features, exposure, n_trials,
                                    overwrite):
        '''Assert the consistency of the inputs dimensions.'''
        validate_1d_array(target)
        size = target.size

        assert x_coords is not None or x_time is not None or x_features is not None

        # Validate arrays with x_variables
        if x_coords is not None:
            validate_2d_array(x_coords, n_rows=size, n_cols=2)

        if x_time is not None:
            validate_1d_array(x_time, size=size)

        if x_features is not None:
            if x_features.ndim == 1:
                validate_1d_array(x_features, size=size)
            else:
                validate_2d_array(x_features, n_cols=None, n_rows=size)

        # Validate other arrays
        if exposure is not None:
            validate_1d_array(exposure, size=size)

        if n_trials is not None:
            validate_1d_array(n_trials, size=size)

        # Store data if requested
        if overwrite:
            self._store_raw_inputs_dims(target=target,
                                        x_coords=x_coords,
                                        x_time=x_time,
                                        x_features=x_features)


    def _validate_prediction_inputs_dims(self, x_coords, x_time, x_features, exposure, n_trials):
        '''Assert the consistency of a new set of inputs for predictions or posterior sampling.'''
        assert x_coords is not None or x_time is not None or x_features is not None

        # Validate arrays with x_variables
        x_size = None

        if self.spatial:
            validate_2d_array(x_coords, n_rows=x_size, n_cols=2)
            x_size = x_coords.shape[0]
        else:
            assert x_coords is None

        if self.temporal:
            validate_1d_array(x_time, size=x_size)
            x_size = x_time.size if x_size is None else x_size
        else:
            assert x_time is None

        if self.n_features == 1:
            validate_1d_array(x_features, size=x_size)
            x_size = x_features.size if x_size is None else x_size
        elif self.n_features > 1:
            validate_2d_array(x_features, n_cols=self.n_features, n_rows=x_size)
            x_size = x_features.shape[0] if x_size is None else x_size
        else:
            assert x_features is None

        # Validate other arrays
        if exposure is not None:
            validate_1d_array(exposure, size=x_size)

        if n_trials is not None:
            validate_1d_array(n_trials, size=x_size)

    def _preprocess_target(self, target):
        '''Preprocessing of target values.'''
        return target


    def _stack_x(self, x_coords=None, x_time=None, x_features=None):
        '''Build X from x_coords, x_time and x_features.'''
        x_list = list()

        # Coords columns
        if x_coords is not None:
            x_list.append(x_coords)

        # Time column
        if x_time is not None:
            x_list.append(x_time[:, None])

        # Feature columns
        if x_features is not None:
            if x_features.ndim == 1:
                x_list.append(x_features[:, None])
            else:
                x_list.append(x_features)

        assert len(set([xi.shape[0] for xi in x_list])) == 1, 'x_variables have different sizes.'

        return np.hstack(x_list)


    def _preprocess_train_x_variables(self, x_coords, x_time, x_features, overwrite=False):
        '''Preprocessing of x_variables for new training.'''
        X = self._stack_x(x_coords=x_coords, x_time=x_time, x_features=x_features)

        # If normalization is required
        if self.x_norm:
            x_norm_rule = self.new_x_norm_rule()
            X = x_norm_rule.fit_transform(X)

            if overwrite: # NOTE: if overwrite is False x_norm_rule is not saved!!!
                self.x_norm_rule = x_norm_rule

        return X


    def _preprocess_prediction_x_variables(self, x_coords, x_time, x_features):
        '''Preprocessing of x_variables for prediction.'''
        X = self._stack_x(x_coords=x_coords, x_time=x_time, x_features=x_features)

        # If normalization is required
        if self.x_norm:
            X = self.x_norm_rule.transform(X)

        return X


    def _build_yxwen(self, target, X, exposure=None, n_trials=None):
        '''Build arrays: y, X, weights, exposure and n_trials to pass to base models.'''
        weights = None
        exposure = exposure
        n_trials = n_trials

        return target, X, weights, exposure, n_trials


    def _fit_base_model(self, y, X, weights, exposure, n_trials):
        '''Train a new instance of the base_model.'''
        raise NotImplementedError


    def _predict_base_model(self, X, exposure, n_trials):
        '''Call the prediction method of the base_model.'''
        raise NotImplementedError


    def _posterior_samples_base_model(self, X, exposure, n_trials, n_samples):
        '''Call the sampling method of the base_model.'''
        raise NotImplementedError


    def fit(self, target, x_coords, x_time=None, x_features=None, exposure=None, n_trials=None,
            overwrite=True):
        '''
        Instantiate a base_model and train it.

        :param target: Variable to predict.
                       Numpy array, shape [n_data, ]
        :param x_coords: Spatial coordiantes associated to the target values (optional).
                         Numpy array, shape [n_data, 2]
        :param x_time: Temporal reference associated to the target values (optional).
                       Numpy array, shape [n_data, ]
        :param x_features: Additional variables/features associated to the target values (optional).
                           Numpy array, shape [n_data, ] or [n_data, n_features]
        :param exposure: For count process models, exposure associated to the target values (optional).
                         Numpy array, shape [n_data, ]
        :param n_trials: For binomial models, number of trials associated to the target values (optional).
                         Numpy array, shape [n_data, ]
        :param overwrite: Whether previous training (if any) will be overwritten.
                          Otherwise the new trained base model is returned (default=True).
                          Boolean object.
        '''
        if overwrite:
            self.n_data = target.size

        # Don't change dims if a frame has been set.
        _overwrite_dims = False if hasattr(self, 'frame') else overwrite

        # Validate inputs
        self._validate_train_inputs_dims(target=target,
                                         x_coords=x_coords,
                                         x_time=x_time,
                                         x_features=x_features,
                                         exposure=exposure,
                                         n_trials=n_trials,
                                         overwrite=_overwrite_dims)

        # Preprocess
        new_target = self._preprocess_target(target=target)
        X = self._preprocess_train_x_variables(x_coords=x_coords,
                                               x_time=x_time,
                                               x_features=x_features,
                                               overwrite=overwrite)

        # Build objects to pass to base model
        y, X, w, e, n = self._build_yxwen(target=new_target, X=X, exposure=exposure, n_trials=n_trials)

        # Train model
        new_base_model = self._fit_base_model(y=y, X=X, weights=w, exposure=e, n_trials=n)

        # Add some summary
        #TODO

        if overwrite:
            self._X_train = X
            self._y_train = y
            self._weights = w
            self._exposure = e
            self._n_trials = n
            self.base_model = new_base_model
        else:
            return new_base_model


    def predict(self, x_coords=None, x_time=None, x_features=None, exposure=None, n_trials=None):
        '''
        Return the predictive mean for a set of new points.

        :param x_coords: Spatial coordiantes associated to the predictions.
                         A numpy array, shape [n_data, 2] or None if model is not spatial.
        :param x_time: Temporal reference associated to the predictions.
                       Numpy array, shape [n_data, ] or None if model is not temporal.
        :param x_features: Additional variables/features associated to the predictions.
                           Numpy array, shape [n_data, ] or [n_data, n_features] or None if n_features = 0.
        :param exposure: For count process models, exposure associated to the predicted values (optional).
                         Numpy array, shape [n_data, ]
        '''
        # Check there is a trained model
        assert hasattr(self, 'base_model'), 'A base_model has not been trained yet.'

        # Validate inputs
        self._validate_prediction_inputs_dims(x_coords=x_coords,
                                              x_time=x_time,
                                              x_features=x_features,
                                              exposure=exposure,
                                              n_trials=n_trials)

        # Preprocess
        X = self._preprocess_prediction_x_variables(x_coords=x_coords,
                                                    x_time=x_time,
                                                    x_features=x_features)

        return self._predict_base_model(X=X, exposure=exposure, n_trials=n_trials)


    def posterior_samples(self, x_coords=None, x_time=None, x_features=None, exposure=None,
                          n_trials=None, n_samples=100):
        '''
        Return samples from the posterior distribution.

        :param x_coords: Spatial coordiantes associated to the sampled values.
                         A numpy array, shape [n_data, 2] or None if model is not spatial.
        :param x_time: Temporal reference associated to the sampled values.
                       Numpy array, shape [n_data, ] or None if model is not temporal.
        :param x_features: Additional variables/features associated to the sampled values.
                           Numpy array, shape [n_data, ] or [n_data, n_features] or None if n_features = 0.
        :param exposure: For count process models, exposure associated to the sampled values (optional).
                         Numpy array, shape [n_data, ]
        :param n_samples: Number of samples to return.
                          Positive integer.
        '''
        assert hasattr(self, 'base_model'), 'A base_model has not been trained yet.'

        # Validate inputs
        self._validate_prediction_inputs_dims(x_coords=x_coords,
                                              x_time=x_time,
                                              x_features=x_features,
                                              exposure=exposure,
                                              n_trials=n_trials)

        # Preprocess
        X = self._preprocess_prediction_x_variables(x_coords=x_coords,
                                                    x_time=x_time,
                                                    x_features=x_features)

        assert isinstance(n_samples, int), 'Expecting a integer number of samples.'
        assert n_samples > 0, 'The number of samples has to be positive.'

        return self._posterior_samples_base_model(X=X, exposure=exposure, n_trials=n_trials,
                                                  n_samples=n_samples)


