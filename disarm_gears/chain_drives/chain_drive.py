import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from ..validators import *


class ChainDrive(object):

    def __init__(self, base_model_gen, x_norm_gen=None):

        assert callable(base_model_gen), 'Object is not callable'
        self.new_base_model = base_model_gen

        # Define x normalizer
        if x_norm_gen is not None:
            assert callable(x_norm_gen), 'Object is not callable'
            assert isinstance(x_norm_gen(), StandardScaler) or isinstance(x_norm_gen(), Normalizer)
            self.x_norm = True
            self.new_x_norm_rule = x_norm_gen
        else:
            self.x_norm = False


    def _store_raw_inputs_dims(self, target, x_coords, x_time, x_features):
        '''Store the inputs dimensions.'''
        self.n_data = target.size
        self.spatial = False if x_coords is None else True
        self.temporal = False if x_time is None else True
        self.n_features = 0 if x_features is None else x_features.shape[1]


    def _validate_train_inputs_dims(self, target, x_coords, x_time, x_features, n_trials, exposure, overwrite):
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
        if n_trials is not None:
            validate_1d_array(n_trials, size=size)

        if exposure is not None:
            validate_1d_array(exposure, size=size)

        # Store data if requested
        if overwrite:
            self._store_raw_inputs_dims(target=target,
                                        x_coords=x_coords,
                                        x_time=x_time,
                                        x_features=x_features)


    def _validate_prediction_inputs_dims(self, x_coords, x_time, x_features, exposure):
        '''Assert the consistency of a new set of inputs for prediction.'''

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

        if exposure is not None:
            validate_1d_array(exposure, size=x_size)


    def _preprocess_target(self, target):
        '''Preprocessing of target values'''
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


    def _build_yxwe(self, target, X, n_trials=None, exposure=None):
        '''Build arrays: y, X, weights and exposure to pass to base models.'''
        weights = None
        exposure = exposure
        return target, X, weights, exposure


    def fit(self, target, x_coords, x_time=None, x_features=None, n_trials=None, exposure=None, overwrite=True):
        '''
        Fit a new model

        :param overwrite: Whether previous training (if any) will be overwritten.
                          Otherwise the new trained base model is returned.
                          Boolean object.
        '''
        # Validate inputs
        self._validate_train_inputs_dims(target=target,
                                         x_coords=x_coords,
                                         x_time=x_time,
                                         x_features=x_features,
                                         n_trials=n_trials,
                                         exposure=exposure,
                                         overwrite=overwrite)

        # Preprocess
        new_target = self._preprocess_target(target=target)
        X = self._preprocess_train_x_variables(x_coords=x_coords,
                                               x_time=x_time,
                                               x_features=x_features,
                                               overwrite=overwrite)

        # Build objects to pass to base model
        y, X, w, e = self._build_yxwe(target=new_target, X=X, n_trials=n_trials, exposure=exposure)

        # Train model
        new_base_model = self._fit_base_model(y=y, X=X, weights=w, exposure=e)

        # Add some summary
        #TODO

        if overwrite:
            self.base_model = new_base_model
        else:
            return new_base_model


    def predict(self, x_coords=None, x_time=None, x_features=None, exposure=None):
        '''
        Make predictions
        '''
        # Check there is a trained model
        assert hasattr(self, 'base_model'), 'A base_model has not been trained yet.'

        # Validate inputs
        self._validate_prediction_inputs_dims(x_coords=x_coords,
                                              x_time=x_time,
                                              x_features=x_features,
                                              exposure=exposure)

        # Preprocess
        X = self._preprocess_prediction_x_variables(x_coords=x_coords,
                                                    x_time=x_time,
                                                    x_features=x_features)

        return self._predict_base_model(X=X, exposure=exposure)


    def posterior_samples(self, x_coords=None, x_time=None, x_features=None, exposure=None, n_samples=100):
        '''
        Generate samples from the posterior distribution
        '''
        # Check there is a trained model
        assert hasattr(self, 'base_model'), 'A base_model has not been trained yet.'

        # Validate inputs
        self._validate_prediction_inputs_dims(x_coords=x_coords,
                                              x_time=x_time,
                                              x_features=x_features,
                                              exposure=exposure)

        # Preprocess
        X = self._preprocess_prediction_x_variables(x_coords=x_coords,
                                                    x_time=x_time,
                                                    x_features=x_features)

        return self._posterior_samples_base_model(X=X, exposure=exposure, n_samples=n_samples)


    def _fit_base_model(self, y, X, weights, exposure):
        raise NotImplementedError


    def _predict_base_model(self, X, exposure):
        raise NotImplementedError


    def _posterior_samples_base_model(self, X, exposure, n_samples):
        raise NotImplementedError
