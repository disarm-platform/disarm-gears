import unittest
import numpy as np
from disarm_gears.chain_drives import ClassificationPipeline as ClassP
from disarm_gears.gears import PrevalenceModel


# Some inputs
_target = np.round(np.random.uniform(0, 4, 3))
_x_coords = np.arange(6).reshape(-1, 2)
_x_time = np.arange(3)
_x_features = np.arange(12).reshape(-1, 4)
_n_trials = np.repeat(5, 3)
_new_x_coords = np.arange(8).reshape(-1, 2)
_new_x_time = np.arange(4)
_new_x_features = np.arange(16).reshape(-1, 4)

def dummy_model(**kwargs):
    return 0

class ClassificationPipelineTests(unittest.TestCase):

    def test_init(self):
        self.assertRaises(NotImplementedError, ClassP, base_model_gen=dummy_model)

    def test_build_yxwe(self):

        pipeline1 = ClassP(base_model_gen=PrevalenceModel, x_norm_gen=None)
        self.assertRaises(AssertionError, pipeline1._build_yxwe, target=_target,
                          X=np.ones_like(_target)[:, None], n_trials=None, exposure=None)
        output1 = pipeline1._build_yxwe(target=_target, X=np.ones_like(_target)[:, None],
                                        n_trials=_n_trials, exposure=None)
        self.assertEqual(len(output1), 4)
        self.assertIsInstance(output1[0], np.ndarray)
        self.assertIsInstance(output1[1], np.ndarray)
        self.assertIsInstance(output1[2], np.ndarray)
        self.assertTrue(output1[3] is None)

    def test_fit(self):

        pipeline1 = ClassP(base_model_gen=PrevalenceModel, x_norm_gen=None)

        # If overwrite == True
        pipeline1.fit(target=_target, x_coords=_x_coords, x_time=_x_time, x_features=_x_features,
                      n_trials=_n_trials, exposure=None, overwrite=True)
        self.assertTrue(hasattr(pipeline1, 'base_model'))

        # If overwrite == False
        bm = pipeline1.fit(target=_target, x_coords=_x_coords, x_time=_x_time,
                           x_features=_x_features, n_trials=_n_trials, exposure=None, overwrite=False)
        self.assertIsInstance(bm, PrevalenceModel)

    def test_predict(self):

        pipeline1 = ClassP(base_model_gen=PrevalenceModel, x_norm_gen=None)
        pipeline1.fit(target=_target, x_coords=_x_coords, x_time=_x_time, x_features=_x_features,
                      n_trials=_n_trials, exposure=None, overwrite=True)
        output1 = pipeline1.predict(x_coords=_new_x_coords, x_time=_new_x_time, x_features=_new_x_features,
                                    exposure=None)
        self.assertIsInstance(output1, np.ndarray)
        self.assertEqual(_new_x_time.size, output1.size)

    def test_posterior_samples(self):

        pipeline1 = ClassP(base_model_gen=PrevalenceModel, x_norm_gen=None)
        pipeline1.fit(target=_target, x_coords=_x_coords, x_time=_x_time, x_features=_x_features,
                      n_trials=_n_trials, exposure=None, overwrite=True)
        output1 = pipeline1.posterior_samples(x_coords=_new_x_coords, x_time=_new_x_time,
                                              x_features=_new_x_features, exposure=None, n_samples=8)
        self.assertIsInstance(output1, np.ndarray)
        self.assertEqual(output1.ndim, 2)
        self.assertEqual(output1.shape[0], 8)
        self.assertEqual(output1.shape[1], _new_x_time.size)

    def test_fit_base_model(self):
        pass

    def test_predict_base_model(self):
        pass

    def test_posterior_samples_base_model(self):
        pass