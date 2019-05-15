import numpy as np
import pandas as pd
import geopandas as geop
from shapely import geometry
from disarm_gears.validators import validate_2d_array, validate_1d_array
from disarm_gears.frames.point_pattern import PointPattern
#from disarm_gears.frames.raster_image import RasterImage TODO
from sklearn.tree import DecisionTreeRegressor
from descartes import PolygonPatch


#NOTE this class is similar to Tesellation, but uses pre-defined polygons as opposed to points
class TilePattern(PointPattern):

    def __init__(self, geometries, attributes=None, crs=None):#'epsg:3857'
        '''
        Define a spatial region based on a GeoDataFrame .

        :param geometries: Geometries that delimit the region.
                           geopandas GeoSeries.
        :param attributes: Attributes associated to the geometries (optional).
                           Numpy array or pandas DataFrame, shape = [n_points, m].
        :param crs: Coordinate reference system (optional).
                    String.
        '''

        assert isinstance(geometries, geop.GeoSeries), 'geometries must be a GeoSeries object.'

        #if crs is not None and crs != region.crs['init']:
        #from functools import partial
        #import pyproj
        #from shapely.ops import transform
        if crs is not None:
            #assert isinstance(crs, str)
            #    project = partial(pyproj.transform,
            #                      pyproj.Proj(init=region.crs['init']),  # source crs
            #                      pyproj.Proj(init=crs))         # destination crs
            #    for i,gi in enumerate(geometries):
            #        geometries[i] = transform(project, gi)
            raise NotImplementedError


        # Define points geometry
        _region = {'geometry': geometries}

        # Add attributes if specified
        if attributes is not None:
            _region = self._add_attributes(region_dict=_region, attributes=attributes)

        self.projection = crs
        self.region = geop.GeoDataFrame(_region, crs=self.projection)
        self.boundary = self.region
        self.centroids = np.vstack(self.region.centroid)

        # Bounding box
        _box = np.asarray(self.region.bounds)
        self.box = pd.DataFrame({'x': (_box[:, [0, 2]].min(), _box[:, [0, 2]].max()),
                                 'y': (_box[:, [1, 3]].min(), _box[:, [1, 3]].max())})


    def set_boundary(self, B):
        '''
        Set a boundary to delimit the spatial region.

        :param B: Polygon(s) that define the region.
                  Geopandas DataFrame.
        '''
        assert isinstance(B, geop.GeoDataFrame)
        self.region = geop.overlay(self.region, B, how='intersection')
        self.boundary = B


    def locate(self, X):
        '''
        Associate geometries in the region with a set of locations X.

        :param X: Set of coordinates.
                  Numpy array, shape = [n, 2]
        :return: Array with tiles per point (-1 for points outside the spatial frame).
                 Numpy array of integers.
        '''
        validate_2d_array(X)
        geom_points = geop.GeoDataFrame(crs=self.projection, geometry=[geometry.Point(xi) for xi in X])
        ix = geop.tools.sjoin(geom_points, self.region, how='left')['index_right']
        ix[np.isnan(ix)] = -1

        return np.array(ix).astype(int)


    def points_to_frame(self, X, group_by=None):
        '''
        From a set of points X, count the number of points per geometry in the region.

        :param X: Set of coordinates.
                  Numpy array, shape = [n, 2].
        :param group_by: Variable that should be used to group X in addition to the geometries (optional).
                         None or numpy array of size n.
        :return: Dataframe with point counts per geometry-group_by.
                 pandas DataFrame.
        '''
        ix = self.locate(X)
        _mask = ix > -1

        group_vars = ['tile']
        _dict = {'tile': ix[_mask], 'var_0': np.ones(_mask.sum())}
        if group_by is not None:
            validate_1d_array(group_by) #TODO define shape in docstring
            _dict.update({'group': group_by[_mask]})
            group_vars.append('group')

        z = pd.DataFrame(_dict)
        z.dropna(inplace=True)
        z = z.groupby(by=group_vars).sum()
        z.reset_index(inplace=True)

        return z


    def marked_points_to_frame(self, X, Y, fun='mean', group_by=None):
        '''
        From a set of values Y associated to a set of locations X,
        compute the statistic fun per geometry in the region.

        :param X: Set of coordinates.
                  Numpy array, shape = [n, 2]
        :param Y: Set of values at locations X.
                  Numpy array of size n
        :param fun: Statistic function to compute per region.
                    String.
        :param group_by: Variable that should be used to group X in addition to the geometries (optional).
                         None or numpy array of size n.
        :return: Dataframe with statistics per geometry-group_by.
                 pandas DataFrame.
        '''
        # Check Y dimensions
        if Y.ndim == 1:
            Y = Y[:, None]
        assert Y.shape[0] == X.shape[0], 'X and Y dimensions do not match.'

        # Define dataframe of variables and tiles
        ix = self.locate(X)
        mask_ = ix > -1
        _dict = {'var_%s' %j: Y[mask_, j] for j in np.arange(Y.shape[1])}

        group_vars = ['tile']
        _dict.update({'tile': ix[mask_]})
        if group_by is not None:
            _dict.update({'group': group_by[mask_]})
            group_vars.append('group')

        z = pd.DataFrame(_dict)
        z.dropna(inplace=True)

        # Group dataframe
        if fun == 'sum':
            #z = z.groupby(by='tile').sum()
            z = z.groupby(by=group_vars).sum()
        elif fun == 'mean':
            #z = z.groupby(by='tile').mean()
            z = z.groupby(by=group_vars).mean()
        else:
            raise NotImplementedError

        z.reset_index(inplace=True)

        return z


    #def raster_to_frame(self, raster, fun='mean', fill_method=None):
    #    '''
    #    From the values in a raster layer, compute the statistic fun per geometry in the region.

    #    :param raster: Raster object.

    #    :param fun: Statistic function to compute per region.
    #                String.
    #    :param fill_method: Method to fill not available values (optional).
    #                        String or None (defaults to None).
    #    :return: Dataframe with statistics per geometry.
    #             pandas DataFrame.
    #    '''
    #    rr = RasterImage(image=raster, thresholds=thresholds) #TODO add object type in docstrings
    #    X = rr.get_coordinates(filter=True)
    #    y = rr.region.ReadAsArray().flatten()
    #    z = self.marked_points_to_frame(X, y, fun=fun)

    #    if fill_method is None:
    #        pass
    #    elif fill_method == 'DecisionTree':
    #        a = np.array(z.index)
    #        b = np.delete(np.array(self.region.index), a)
    #        if b.size > 0:
    #            Xa = X[a]
    #            Xb = X[b]
    #            Xa = np.hstack([Xa, (Xa[:, 0] * Xa[:, 1])[:, None]])
    #            Xb = np.hstack([Xb, (Xb[:, 0] * Xb[:, 1])[:, None]])
    #            ya = z['var_0']
    #            m = DecisionTreeRegressor()
    #            m.fit(Xa, ya)
    #            yb = m.predict(Xb)
    #            z = pd.DataFrame(data={'var_0': np.hstack([ya, yb])}, index=np.hstack([a, b]))
    #            z.sort_index(inplace=True)
    #    else:
    #        #TODO: Add other methods to fill missing values
    #        raise NotImplementedError

    #    return z



#def _get_geometry_grid(self, size=40):
#    return np.meshgrid(np.linspace(*self.box['x'], size), np.linspace(*self.box['y'], size))


