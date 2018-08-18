import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from disarm_gears.chain_drives import SupervisedLearningCore as SLC


# Some inputs
_target = np.ones(3)
_x_coords = np.arange(6).reshape(-1, 2)
_x_time = np.arange(3)
_x_features = np.arange(12).reshape(-1, 4)
_new_x_coords = np.arange(8).reshape(-1, 2)
_new_x_time = np.arange(4)
_new_x_features = np.arange(16).reshape(-1, 4)

# Dummy model
def dummy_base_model(**kwargs):
    return 0


class SupervisedLearningCoreTests(unittest.TestCase):

    def test_init_core_class(self):

        # Check bad inputs
        self.assertRaises(AssertionError, SLC, base_model_gen=None, x_norm_gen=True)
        self.assertRaises(AssertionError, SLC, base_model_gen=lambda x:x, x_norm_gen=True)

        # Check attributes after init
        pipeline1 = SLC(base_model_gen=lambda x:x, x_norm_gen=None)
        pipeline2 = SLC(base_model_gen=lambda x:x, x_norm_gen=StandardScaler)
        self.assertTrue(hasattr(pipeline1, 'new_base_model'))
        self.assertTrue(hasattr(pipeline1, 'x_norm'))
        self.assertTrue(hasattr(pipeline2, 'new_x_norm_rule'))

    def test_store_raw_inputs_dims(self):

        pipeline1 = SLC(base_model_gen=lambda x:x, x_norm_gen=None)

        # All x_variables
        pipeline1._store_raw_inputs_dims(target=_target, x_coords=_x_coords, x_time=_x_time,
                                         x_features=_x_features)
        self.assertEqual(pipeline1.n_data, 3)
        self.assertTrue(pipeline1.spatial)
        self.assertTrue(pipeline1.temporal)
        self.assertEqual(pipeline1.n_features, 4)

        # Not spatial
        pipeline1._store_raw_inputs_dims(target=_target, x_coords=None, x_time=_x_time,
                                         x_features=_x_features)
        self.assertFalse(pipeline1.spatial)

        # Only spatial
        pipeline1._store_raw_inputs_dims(target=_target, x_coords=_x_coords, x_time=None,
                                         x_features=None)
        self.assertFalse(pipeline1.temporal)
        self.assertEquals(pipeline1.n_features, 0)

    def test_validate_train_inputs_dims(self):

        pipeline1 = SLC(base_model_gen=lambda x:x, x_norm_gen=None)
        bad_target = np.ones(5)

        self.assertRaises(AssertionError, pipeline1._validate_train_inputs_dims,
                          target=bad_target, x_coords=None, x_time=None, x_features=None,
                          n_trials=None, exposure=None, overwrite=False)
        self.assertRaises(AssertionError, pipeline1._validate_train_inputs_dims,
                          target=bad_target, x_coords=_x_coords, x_time=None, x_features=None,
                          n_trials=None, exposure=None, overwrite=False)
        self.assertRaises(AssertionError, pipeline1._validate_train_inputs_dims,
                          target=bad_target, x_coords=None, x_time=_x_time, x_features=None,
                          n_trials=None, exposure=None, overwrite=False)
        self.assertRaises(AssertionError, pipeline1._validate_train_inputs_dims,
                          target=bad_target, x_coords=None, x_time=_x_time, x_features=_x_features,
                          n_trials=None, exposure=None, overwrite=False)

    def test_validate_prediction_inputs_dims(self):

        pipeline1 = SLC(base_model_gen=lambda x:x, x_norm_gen=None)

        # All x_variables
        pipeline1._store_raw_inputs_dims(target=_target,x_coords=_x_coords, x_time=_x_time,
                                         x_features=_x_features)
        self.assertRaises(AssertionError, pipeline1._validate_prediction_inputs_dims,
                          x_coords=_new_x_coords, x_time=_new_x_time, x_features=None,
                          exposure=None, n_trials=None)
        self.assertRaises(AssertionError, pipeline1._validate_prediction_inputs_dims,
                          x_coords=_new_x_coords, x_time=None, x_features=_new_x_features,
                          exposure=None, n_trials=None)
        self.assertRaises(AssertionError, pipeline1._validate_prediction_inputs_dims,
                          x_coords=None, x_time=_x_time, x_features=_new_x_features,
                          exposure=None, n_trials=None)

        # Not spatial
        pipeline1._store_raw_inputs_dims(target=_target,x_coords=None, x_time=_x_time,
                                         x_features=_x_features)
        self.assertRaises(AssertionError, pipeline1._validate_prediction_inputs_dims,
                          x_coords=_new_x_coords, x_time=_new_x_time, x_features=_new_x_features,
                          exposure=None, n_trials=None)

        # Not temporal
        pipeline1._store_raw_inputs_dims(target=_target,x_coords=None, x_time=None,
                                         x_features=_x_features)
        self.assertRaises(AssertionError, pipeline1._validate_prediction_inputs_dims,
                          x_coords=None, x_time=_new_x_time, x_features=_new_x_features,
                          exposure=None, n_trials=None)

        # No features
        pipeline1._store_raw_inputs_dims(target=_target,x_coords=_x_coords, x_time=_x_time,
                                         x_features=None)
        self.assertRaises(AssertionError, pipeline1._validate_prediction_inputs_dims,
                          x_coords=_new_x_coords, x_time=_new_x_time, x_features=_new_x_features,
                          exposure=None, n_trials=None)

    def test_stack_x(self):

        pipeline1 = SLC(base_model_gen=lambda x:x, x_norm_gen=None)
        bad_x_coords = np.arange(8).reshape(-1, 2)
        self.assertRaises(AssertionError, pipeline1._stack_x, x_coords=bad_x_coords,
                          x_time=_x_time,x_features=_x_features)

    def test_preprocess_target(self):

        pipeline1 = SLC(base_model_gen=lambda x:x, x_norm_gen=None)
        new_target = pipeline1._preprocess_target(target=_target)
        self.assertIsInstance(new_target, np.ndarray)

    def test_preprocess_train_x_variables(self):

        pipeline1 = SLC(base_model_gen=lambda x:x, x_norm_gen=None)
        X = pipeline1._preprocess_train_x_variables(x_coords=_x_coords, x_time=_x_time,
                                                    x_features=_x_features)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape[1], _x_coords.shape[1] + _x_features.shape[1] + 1)

    def test_preprocess_prediction_x_variables(self):

        pipeline1 = SLC(base_model_gen=lambda x:x, x_norm_gen=None)
        X = pipeline1._preprocess_prediction_x_variables(x_coords=_x_coords, x_time=_x_time,
                                                    x_features=_x_features)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape[1], _x_coords.shape[1] + _x_features.shape[1] + 1)


# The tests below should be implemented for all subclasses as well
class SLCSubclassTests(unittest.TestCase):

    def test_init(self):
        pass

    def test_build_yxwen(self):

        pipeline1 = SLC(base_model_gen=lambda x:x, x_norm_gen=None)
        output1 = pipeline1._build_yxwen(target=_target, X=np.ones_like(_target)[:,None],
                                        exposure=None, n_trials=None)
        self.assertEqual(len(output1), 5)
        self.assertTrue(isinstance(output1[0], np.ndarray))
        self.assertTrue(output1[1] is None or isinstance(output1[1], np.ndarray))
        self.assertTrue(output1[2] is None or isinstance(output1[2], np.ndarray))
        self.assertTrue(output1[3] is None or isinstance(output1[3], np.ndarray))

    def test_fit(self):

        pipeline1 = SLC(base_model_gen=lambda x:x, x_norm_gen=None)
        self.assertRaises(NotImplementedError, pipeline1.fit, target=_target, x_coords=_x_coords,
                          x_time=_x_time, x_features=_x_features, n_trials=None, exposure=None,
                          overwrite=False)
        self.assertRaises(NotImplementedError, pipeline1.fit, target=_target, x_coords=_x_coords,
                          x_time=_x_time, x_features=_x_features, n_trials=None, exposure=None,
                          overwrite=True)

    def test_predict(self):

        pipeline1 = SLC(base_model_gen=lambda x:x, x_norm_gen=None)

        # Prediction without previous training
        self.assertRaises(AssertionError, pipeline1.predict, x_coords=_x_coords,
                          x_time=_x_time, x_features=_x_features, exposure=None)

        # Fake training
        pipeline1._store_raw_inputs_dims(target=_target, x_coords=_x_coords, x_time=_x_time,
                                         x_features=_x_features)
        pipeline1.base_model = dummy_base_model()

        self.assertRaises(NotImplementedError, pipeline1.predict, x_coords=_x_coords,
                          x_time=_x_time, x_features=_x_features, exposure=None)

    def test_posterior_samples(self):

        pipeline1 = SLC(base_model_gen=lambda x:x, x_norm_gen=None)

        # Sampling without previous training
        self.assertRaises(AssertionError, pipeline1.posterior_samples, x_coords=_x_coords,
                          x_time=_x_time, x_features=_x_features, exposure=None)

        # Fake training
        pipeline1._store_raw_inputs_dims(target=_target, x_coords=_x_coords, x_time=_x_time,
                                         x_features=_x_features)
        pipeline1.base_model = dummy_base_model()

        self.assertRaises(NotImplementedError, pipeline1.posterior_samples, x_coords=_x_coords,
                          x_time=_x_time, x_features=_x_features, exposure=None)

    def test_fit_base_model(self):
        pass

    def test_predict_base_model(self):
        pass

    def test_posterior_samples_base_model(self):
        pass
