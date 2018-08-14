import unittest
import numpy as np
from disarm_gears.chain_drives import ClassificationPipeline as ClassP


# Some inputs
_target = np.ones(3)
_x_coords = np.arange(6).reshape(-1, 2)
_x_time = np.arange(3)
_x_features = np.arange(12).reshape(-1, 4)
_new_x_coords = np.arange(8).reshape(-1, 2)
_new_x_time = np.arange(4)
_new_x_features = np.arange(16).reshape(-1, 4)

def dummy_model(**kwargs):
    return 0

class ClassificationPipelineTests(unittest.TestCase):

    def test_init(self):
        self.assertRaises(NotImplementedError, ClassP, base_model_gen=dummy_model)

'''
    def test_build_yxwe(self):

        pipeline1 = ClassP(base_model_gen=lambda x:x, x_norm_gen=None)
        output1 = pipeline1._build_yxwe(target=_target, X=np.ones_like(_target)[:,None],
                                        n_trials=None, exposure=None)
        self.assertEqual(len(output1), 4)
        self.assertTrue(isinstance(output1[0], np.ndarray))
        self.assertTrue(output1[1] is None or isinstance(output1[1], np.ndarray))
        self.assertTrue(output1[2] is None or isinstance(output1[2], np.ndarray))
        self.assertTrue(output1[3] is None or isinstance(output1[3], np.ndarray))

    def test_fit(self):

        pipeline1 = ClassP(base_model_gen=lambda x:x, x_norm_gen=None)
        self.assertRaises(NotImplementedError, pipeline1.fit, target=_target, x_coords=_x_coords,
                          x_time=_x_time, x_features=_x_features, n_trials=None, exposure=None,
                          overwrite=False)
        self.assertRaises(NotImplementedError, pipeline1.fit, target=_target, x_coords=_x_coords,
                          x_time=_x_time, x_features=_x_features, n_trials=None, exposure=None,
                          overwrite=True)

    def test_predict(self):

        pipeline1 = ClassP(base_model_gen=lambda x:x, x_norm_gen=None)
        self.assertRaises(AssertionError, pipeline1.predict, x_coords=_x_coords,
                          x_time=_x_time, x_features=_x_features, exposure=None)

        # Fake training
        pipeline1._store_raw_inputs_dims(target=_target, x_coords=_x_coords, x_time=_x_time,
                                         x_features=_x_features)
        pipeline1.base_model = dummy_base_model()

        self.assertRaises(NotImplementedError, pipeline1.predict, x_coords=_x_coords,
                          x_time=_x_time, x_features=_x_features, exposure=None)

    def test_posterior_samples(self):

        pipeline1 = ClassP(base_model_gen=lambda x:x, x_norm_gen=None)
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
'''
