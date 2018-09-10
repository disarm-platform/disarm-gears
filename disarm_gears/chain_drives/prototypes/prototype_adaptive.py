import numpy as np
import json
import pygam
from disarm_gears.util import binomial_to_bernoulli, trend_2nd_order
from disarm_gears.frames import Tessellation
from disarm_gears.validators import *


def adaptive_prototype_0(x_coords, n_positive, n_trials, x_id=None, threshold=.5):

    # Validate inputs
    validate_2d_array(x_coords, n_cols=2)
    _size = x_coords.shape[0]
    validate_1d_array(n_positive, size=_size)
    validate_non_negative_array(n_positive)
    validate_integer_array(n_positive)
    validate_positive_array(n_trials)
    validate_integer_array(n_trials)
    validate_1d_array(n_trials, size=_size)
    assert isinstance(threshold, float)

    # Define tessellation
    if x_id is None:
        x_id = np.arange(_size)
    else:
        validate_1d_array(x_id, size=_size)

    ts = Tessellation(x_coords)
    ts_export = {id: {'lng': zi.boundary.coords.xy[0].tolist(),
                      'lat': zi.boundary.coords.xy[1].tolist()}
                 for zi in ts.region.geometry for id in x_id}


    # Preprocess data
    target, weights, X = binomial_to_bernoulli(n_positive=n_positive, n_trials=n_trials,
                                               X=x_coords, aggregated=True)
    new_X = trend_2nd_order(X)

    # Train model
    base_model = pygam.LogisticGAM()
    base_model.gridsearch(y=target, X=new_X, weights=weights)

    n_samples = 300
    new_x_coords = trend_2nd_order(x_coords)
    m_simulations = base_model.sample(X=new_X, y=target, weights=weights, sample_at_X=new_x_coords,
                                      quantity='mu', n_draws=n_samples)

    m_prev = m_simulations.mean(0)
    m_prob = (m_simulations > threshold).sum(0) / n_samples
    m_category = np.zeros_like(m_prob)
    m_category[m_prev > threshold] = 1
    entropy = (- m_prob * np.log2(m_prob) - (1-m_prob) * np.log2(1 - m_prob))
    entropy[np.isnan(entropy)] = 0

    m_export = {'id': x_id.tolist(), 'prevalence': m_prev.tolist(), 'category': m_category.tolist(),
                'entropy': entropy.tolist()}

    joint_output = {'polygons': ts_export, 'estimates': m_export}

    return joint_output
