from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import geopandas as geop
import gdal
import pygam
from disarm_gears.frames import Geometry, Timeframe
from sklearn.cluster import DBSCAN
from disarm_gears.util.surface_trends import trend_2nd_order

# Initial data
#end_date = '2017-06-20'

def dchs(hs_centroid, hs_size, X):
    '''Compute the distances form hot_spots to all locations in X.'''
    dist = np.vstack([np.sqrt((X[:, 0] - hj[0]) ** 2. + (X[:, 1] - hj[1]) ** 2.) for hj in hs_centroid]).T
    argm = np.argmin(dist, axis=1)
    return np.exp(-np.min(dist, axis=1)), hs_size[argm]


def sentinel(end_date, dynamic_data, storage_path):
    '''Forecast incidence of malaria per village.'''
    # Define timeframe
    tf_obsv = Timeframe(start=None, end=end_date, length=6, by='day', step=28)
    tf = Timeframe(start=tf_obsv.start, length=7, by='day', step=28)
    max_knot = tf.knots_info['knot'].max()

    ## Static data processing

    # Load villages polygons
    vfile = Path(storage_path, 'tha_villages.file')
    if vfile.is_file():
        # Load binary file if it exists
        with open(str(vfile.resolve()), 'rb') as f:
            rv = pickle.load(f)
    else:
        # Create Geometry object if binary file is missing
        vfile = Path(storage_path, 'tha_villages/tha_villages.shp')
        assert vfile.is_file(), 'Shapefile with villages could not be found.'
        villages = geop.GeoDataFrame.from_file(str(vfile.resolve()))
        rv = Geometry(villages)
        rv.centroids = np.asarray(rv.region[['lng', 'lat']])
        # Save binary
        with open(storage_path + 'tha_villages.file', 'wb') as f:
            pickle.dump(rv, f, pickle.HIGHEST_PROTOCOL)

    # Load elevation data
    efile = Path(storage_path, 'tha_elevation.file')
    if efile.is_file():
        # Load binary file if it exists
        with open(str(efile.resolve()), 'rb') as f:
            _elevation = pickle.load(f)
    else:
        efile = Path(storage_path, 'tha_elevation.tif')
        assert efile.is_file(), 'Raster file with elevation could not be found.'
        _elevation = rv.raster_to_frame(gdal.Open(str(efile.resolve())), thresholds=[-100, None],
                                        fill_missing=True)
        # Save binary
        with open(storage_path + 'tha_elevation.file', 'wb') as f:
            pickle.dump(_elevation, f, pickle.HIGHEST_PROTOCOL)

    # Merge static data into the same dataframe
    sfile = Path(storage_path, 'tha_static.csv')
    assert sfile.is_file(), 'Static data file could not be found.'
    static_data = pd.read_csv(str(sfile.resolve()))
    x_static = np.array(static_data[['lng', 'lat']])
    y_static = np.array(static_data[['Total_pop', 'Total.ITN', 'Total.IRS']])
    _static = rv.marked_points_to_frame(x_static, y_static, fun='mean')
    villages = rv.region[['lng', 'lat']]
    villages['elevation'] = _elevation.iloc[villages.index]
    villages['pop'] = _static.iloc[villages.index]['var_0']
    villages['itn'] = _static.iloc[villages.index]['var_1']
    villages['irs'] = _static.iloc[villages.index]['var_2']

    ## Dynamic data processing
    dynamic_data['date'] = pd.to_datetime(dynamic_data['date'])
    dynamic_data = dynamic_data.loc[np.logical_and(dynamic_data['date'] >= tf.start,
                                                   dynamic_data['date'] <= tf.end)]
    dynamic_data = dynamic_data.assign(knot=tf.which_knot(np.array(dynamic_data['date'])))
    dynamic_data = dynamic_data.assign(tile=rv.locate(np.array(dynamic_data[['lng', 'lat']]))[:, None])
    dynamic_data.drop(labels=['lng', 'lat'], axis=1, inplace=True)
    dynamic_data = dynamic_data.groupby(by=['knot', 'tile']).sum()

    ## Compute incidence for each knot
    xy_ = villages[['lng', 'lat', 'elevation', 'pop', 'itn', 'irs']].copy()
    xy_.loc[:, 'total_cases'] = 0
    xy_.loc[:, 'imported_cases'] = 0
    xy_.loc[:, 'hs_distance'] = 0
    xy_.loc[:, 'hs_size'] = 0

    XY = pd.DataFrame({})
    exceptions = []
    for k in range(1, max_knot + 1):
        # Copy static data
        xy_k = xy_.copy()
        xy_k['knot'] = k
        # Current data
        if k < max_knot:
            if k in dynamic_data.index.get_level_values(level=0):
                dynamic_k = dynamic_data.xs(k, level='knot').copy()
                xy_k.loc[dynamic_k.index, 'total_cases'] = dynamic_k['total_cases']
        else:
            xy_k.loc[:, 'total_cases'] = np.nan
        # Lagged data
        if k-1 in dynamic_data.index.get_level_values(level=0):
            dynamic_k_lag = dynamic_data.xs(k-1, level='knot').copy()
            xy_k.loc[dynamic_k_lag.index, 'imported_cases'] = dynamic_k_lag['imported_cases']
            # Find clusters of lagged cases
            _ix = dynamic_k_lag.index.get_values()[dynamic_k_lag['total_cases'] > 0]
            x_positive = xy_k.loc[_ix, ['lng', 'lat']]
            db = DBSCAN(eps=.35, min_samples=4, algorithm='auto').fit(x_positive)
            hs = np.vstack([np.array(x_positive.loc[db.labels_ == li, ['lng', 'lat']]).mean(0)
                            for li in np.arange(db.labels_.max() + 1)])
            hs_size = np.array([sum(db.labels_ == li) for li in range(db.labels_.max() + 1)])
            # Make X_k
            hs_distance, hs_size = dchs(hs_centroid=hs, hs_size=hs_size, X=np.array(xy_[['lng', 'lat']]))
            xy_k.loc[:, 'hs_distance'] = hs_distance
            xy_k.loc[:, 'hs_size'] = np.log(hs_size)
        else:
            exceptions.append(k-1)

        XY = pd.concat([XY, xy_k])

    # Define Poisson model and predict incidence
    mask_obsv = XY['knot'] < max_knot
    mask_fore = XY['knot'] == max_knot
    mask_past = XY['knot'] == max_knot - 1
    _surf = False
    if _surf:
        gps = ['lng', 'lat']
        covariates = ['elevation', 'itn', 'irs', 'imported_cases', 'hs_distance', 'hs_size']
        X_obsv = np.hstack([trend_2nd_order(np.array(XY.loc[mask_obsv, gps])),
                            np.array(XY.loc[mask_obsv, covariates])])
        X_fore = np.hstack([trend_2nd_order(np.array(XY.loc[mask_fore, gps])),
                            np.array(XY.loc[mask_fore, covariates])])
        X_past = np.hstack([trend_2nd_order(np.array(XY.loc[mask_past, gps])),
                            np.array(XY.loc[mask_past, covariates])])
    else:
        covariates = ['lng', 'lat', 'elevation', 'itn', 'irs', 'imported_cases', 'hs_distance', 'hs_size']
        X_obsv = np.array(XY.loc[mask_obsv, covariates])
        X_fore = np.array(XY.loc[mask_fore, covariates])
        X_past = np.array(XY.loc[mask_past, covariates])

    y_obsv = np.array(XY.loc[mask_obsv, 'total_cases'])
    exposure_obsv = np.array(XY.loc[mask_obsv, 'pop'])
    #pop_fore = np.array(XY.loc[mask_fore, 'pop'])

    #cases_monthly = np.array([np.sum(XY.loc[XY['knot'] == ki, 'total_cases']) for ki in range(1, max_knot)])
    #max_diff = np.diff(cases_monthly).max()
    # Model
    n_samples = 100

    if _surf:
        gam = pygam.PoissonGAM(pygam.s(0) + pygam.s(1) + pygam.s(2) + pygam.s(3) + pygam.s(4) +
                               pygam.s(5) + pygam.s(6) + pygam.s(7) + pygam.s(8, by=9), lam=100)
    else:
        gam = pygam.PoissonGAM(pygam.te(0, 1) + pygam.s(2) + pygam.s(3) +
                               pygam.s(4) + pygam.s(5) + pygam.s(6, by=7), lam=100.)
    gam.fit(X=X_obsv, y=y_obsv, exposure=exposure_obsv)

    samples_fore = gam.sample(X=X_obsv, y=y_obsv, sample_at_X=X_fore, n_draws=n_samples)

    # Forecast mean
    incidence_hat = gam.predict(X_fore)

    # Exceedance prob
    incidence_past = gam.predict(X_past)
    exceedance_prob = np.sum(samples_fore - incidence_past > 0, axis=0) / n_samples

    # Classificatons
    incidence_perc = np.percentile(incidence_hat, q=[50., 75., 90., 95.])
    exceedance_perc = np.percentile(exceedance_prob, q=[50., 75., 90., 95.])

    incidence_class = np.zeros_like(incidence_hat)
    exceedance_class = np.zeros_like(incidence_hat)

    for i in range(4):
        incidence_class[incidence_hat > incidence_perc[i]] = i + 1
        exceedance_class[exceedance_prob > exceedance_perc[i]] = i + 1

    XY_obsv = XY.loc[mask_obsv]
    XY_obsv.loc[:, 'total_incidence'] = gam.predict(X_obsv)
    XY_fore = XY.loc[mask_fore]
    XY_fore.loc[:, 'total_incidence'] = incidence_hat
    XY_fore.loc[:, 'exceedance_prob'] = exceedance_prob
    XY_fore.loc[:, 'total_incidence_class'] = incidence_class
    XY_fore.loc[:, 'exceedance_class'] = exceedance_class

    return XY_obsv, XY_fore, gam

