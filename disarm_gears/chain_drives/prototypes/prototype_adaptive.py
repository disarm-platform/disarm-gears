import numpy as np
import json
import pygam
from disarm_gears.util import binomial_to_bernoulli, trend_2nd_order
from disarm_gears.frames import Tessellation
from disarm_gears.validators import *

#To call Algorithmia
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def df_to_geojson(df, layer_names=[], lat='lat', lng='lng'): #TODO move this function to util?
    '''Put dataframe into geojson format'''
    geojson = {'type': 'FeatureCollection', 'features': []}
    for _, row in df.iterrows():
        _feature = {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'Point', 'coordinates': []}}
        _feature['geometry']['coordinates'] = [row[lng],row[lat]]
        geojson['features'].append(_feature)

    json = {'layer_names': layer_names, 'points': geojson}
    return json


def adaptive_prototype_0(x_frame, x_id, x_coords, n_positive, n_trials,
                         threshold=.5, covariate_layers=np.array([1, 4, 12, 15]), random_seed=123):

    # Set random seed
    np.random.seed(random_seed)

    # Validate inputs
    ## x_frame and x_id
    validate_2d_array(x_frame, n_cols=2)
    frame_size = x_frame.shape[0]
    if x_id is None:
        x_id = np.arange(frame_size)
    else:
        validate_1d_array(x_id, size=frame_size)
    ## Training data
    validate_2d_array(x_coords, n_cols=2)
    train_size = x_coords.shape[0]
    validate_1d_array(n_positive, size=train_size)
    validate_non_negative_array(n_positive)
    validate_integer_array(n_positive)
    validate_positive_array(n_trials)
    validate_integer_array(n_trials)
    validate_1d_array(n_trials, size=train_size)

    ## Request parameters
    assert isinstance(threshold, float)
    if covariate_layers is not None:
        validate_1d_array(covariate_layers) #TODO: check covariates are valid

    # Define tessellation
    ts = Tessellation(x_frame)
    ts_export = {id: {'lng': zi.boundary.coords.xy[0].tolist(),
                      'lat': zi.boundary.coords.xy[1].tolist()}
                 for zi, id in zip(ts.region.geometry, x_id)}

    # Preprocess data
    target, weights, X = binomial_to_bernoulli(n_positive=n_positive, n_trials=n_trials,
                                               X=x_coords, aggregated=True)

    # Find covariates
    if covariate_layers is None:
        new_X = trend_2nd_order(X)
        new_x_coords = trend_2nd_order(x_frame)

    else:
        layer_names = ['bioclim%s' %j for j in covariate_layers]
        layer_names += ['elev_m', 'dist_to_water_m']
        x_frame_js = df_to_geojson(pd.DataFrame(x_frame, columns=['lng', 'lat']), layer_names=layer_names)
        x_train_js = df_to_geojson(pd.DataFrame(x_coords, columns=['lng', 'lat']), layer_names=layer_names)
        algo_link = 'http://faas.srv.disarm.io/function/fn-covariate-extractor'
        algo_frame = requests.post(algo_link, data=json.dumps(x_frame_js))
        algo_train = requests.post(algo_link, data=json.dumps(x_train_js))
        cov_frame = np.vstack([[js['properties'][k] for k in layer_names] for js in algo_frame.json()['result']['features']])
        cov_train = np.vstack([[js['properties'][k] for k in layer_names] for js in algo_train.json()['result']['features']])


        # Fit ML models
        _t, _w, ml_train_X = binomial_to_bernoulli(n_positive=n_positive, n_trials=n_trials,
                                                   X=cov_train, aggregated=True)


        rf = RandomForestClassifier(max_depth=10, random_state=0)
        rf.fit(X=ml_train_X, y=target)
        ml_train = rf.predict_proba(ml_train_X)[:, 1:]
        ml_frame = rf.predict_proba(cov_frame)[:, 1:]

        new_X = np.hstack([trend_2nd_order(X), ml_train])
        new_x_coords = np.hstack([trend_2nd_order(x_frame), ml_frame])


    # Train model
    base_model = pygam.LogisticGAM()
    base_model.gridsearch(y=target, X=new_X, weights=weights)

    n_samples = 200
    m_simulations = base_model.sample(X=new_X, y=target, weights=weights, sample_at_X=new_x_coords,
                                      quantity='mu', n_draws=n_samples)

    #m_prev = m_simulations.mean(0)
    m_prev = base_model.predict_mu(new_x_coords)
    m_prob = (m_simulations > threshold).sum(0) / n_samples

    m_category = np.zeros_like(m_prob)
    m_category[m_prev > threshold] = 1

    entropy = (- m_prob * np.log2(m_prob) - (1-m_prob) * np.log2(1 - m_prob))
    entropy[np.isnan(entropy)] = 0

    #m_export = {'id': x_id.tolist(), 'prevalence': m_prev.tolist(), 'category': m_category.tolist(),
    #            'entropy': entropy.tolist()}
    m_export = {'id': x_id.tolist(), 'exceedance_prob': m_prob.tolist(), 'category': m_category.tolist(),
                'entropy': entropy.tolist()}

    joint_output = {'polygons': ts_export, 'estimates': m_export}

    return joint_output
