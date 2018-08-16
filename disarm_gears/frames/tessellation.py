import numpy as np
import pandas as pd
import geopandas as geop
from scipy.spatial import Voronoi
from shapely import geometry
from shapely.ops import polygonize
from ..validators import validate_2d_array


class Tessellation:

    def __init__(self, points, attributes=None, crs=None):
        '''
        Define a tessellation frame based on a set of points.

        :param points: Set of coordinates.
                       Numpy array, shape = [n_points, 2].
        :param attributes: Attributes associated to the points (optional).
                           Numpy array or pandas DataFrame, shape = [n_points, m].
        :param crs: Coordinate refrence system (optional).
                    String.
        '''
        # NOTE: The use of crs is not tested yet!!!

        # Validate input points
        if isinstance(points, pd.DataFrame):
            points = np.array(points)
        validate_2d_array(points, n_cols=2)

        if crs is not None:
            assert isinstance(crs, str)

        self.centroids = points

        # Voronoi tessellation based on points
        n_points = points.shape[0]

        c1, c2 = np.sort(points[:, 0]), np.sort(points[:, 1])
        _diffs = np.array([np.diff(c1).mean(), np.diff(c2).mean()])

        min_c1, min_c2 = points.min(0) - _diffs
        max_c1, max_c2 = points.max(0) + _diffs

        extra_points = np.vstack([np.vstack([np.repeat(min_c1, n_points), c2]).T,
                                  np.vstack([np.repeat(max_c1, n_points), c2]).T,
                                  np.vstack([c1, np.repeat(min_c2, n_points)]).T,
                                  np.vstack([c1, np.repeat(max_c2, n_points)]).T])

        _points = np.vstack([points, extra_points])
        vor = Voronoi(_points)

        # Define polygons geometry based on tessellation
        lines = [geometry.LineString(vor.vertices[li]) for li in vor.ridge_vertices if -1 not in li]
        disord = geometry.MultiPolygon(list(polygonize(lines)))
        ix_order = np.array([[i for i, di in enumerate(disord) if di.contains(geometry.Point(pi))]
                    for pi in points]).ravel()

        _region = {'geometry': geometry.MultiPolygon([disord[i] for i in ix_order])}

        # Add attributes if specified
        if attributes is not None:

            if isinstance(attributes, pd.DataFrame):
                assert attributes.shape[0] == n_points, 'Attributes and points dimensions do not match.'
                _region.update({'%s' %ai: attributes[ai] for ai in attributes.columns})

            elif isinstance(attributes, np.ndarray):
                if attributes.ndim == 1:
                    assert attributes.size == n_points, 'Attributes and points dimensions do not match.'
                    n_attributes = 1
                    _region.update({'var_0:': attributes})
                elif attributes.ndim == 2:
                    assert attributes.shape[0] == n_points, 'Attributes and points dimensions do not match.'
                    n_attributes = attributes.shape[1]
                    _region.update({'var_%s:' %i: ai for i, ai in enumerate(attributes.T)})
                else:
                    raise ValueError('attributes dimensions not understood.')
                self.slice_attributes = slice(1, 1 + n_attributes)

            else:
                ValueError('attributes type not understood.')

        # Define region as GeoDataFrame
        self.projection = crs
        self.region = geop.GeoDataFrame(_region, crs=self.projection)
        self.region.index.name = None

        # Bounding box
        box_ = np.asarray(_region['geometry'].bounds)
        self.box = pd.DataFrame({'x': box_[[0, 2]], 'y': box_[[1, 3]]})

    def attributes_array(self):
        #TODO add test
        return np.array(self.region.iloc[:, self.slice_attributes])


    def locate(self, X):
        '''
        Identify in the tiles in which a set of points X are located.

        :param X: Set of coordinates.
                  Numpy array, shape = [n, 2]
        '''
        # Validate input
        validate_2d_array(X, n_cols=2)

        # Build GeoDataFrame of X and join it with self.region
        geom_points = geop.GeoDataFrame(crs=self.projection, geometry=[geometry.Point(xi) for xi in X])
        ix = geop.tools.sjoin(geom_points, self.region, how='left')['index_right']
        ix[np.isnan(ix)] = -1

        return np.array(ix).astype(int)
