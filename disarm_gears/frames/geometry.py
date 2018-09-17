import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
from shapely import geometry
from disarm_gears.validators import validate_2d_array
from descartes import PolygonPatch
from functools import partial
import pyproj
from shapely.ops import transform
from disarm_gears.frames import RasterImage

from sklearn.tree import DecisionTreeRegressor


class Geometry:

    def __init__(self, region, crs=None):#'epsg:3857'
        '''
        Define a region based on a GeoDataFrame .

        :param points: Set of coordinates.
                       Numpy array, shape = [n_points, 2].
        :param attributes: Attributes associated to the points (optional).
                           Numpy array or pandas DataFrame, shape = [n_points, m].
        :param crs: Coordinate refrence system (optional).
                    String.
        '''

        assert isinstance(region, gp.GeoDataFrame)
        #assert np.all([isinstance(ri, geometry.Polygon) for ri in region.geometry])

        if crs is not None and crs != region.crs['init']:
            #TODO fixme
            project = partial(pyproj.transform,
                              pyproj.Proj(init=region.crs['init']),  # source crs
                              pyproj.Proj(init=crs))         # destination crs

            for i,gi in enumerate(region.geometry):
                region.geometry[i] = transform(project, gi)


        self.region = region
        self.centroids = np.vstack(self.region.centroid)

        # Coordinate reference system
        self.projection = self.region.crs

        # Bounding box
        box_ = np.asarray(self.region.bounds)
        self.box = pd.DataFrame({'x': (box_[:, [0, 2]].min(), box_[:, [0, 2]].max()),
                                 'y': (box_[:, [1, 3]].min(), box_[:, [1, 3]].max())})


    def locate(self, X):
        '''
        Identify in the tiles in which a set of points X are located.

        :param X: Set of coordinates.
                  Numpy array, shape = [n, 2]
        :return: Array with tiles per point (-1 for points outside the spatial frame).
                 Numpy array of integers.
        '''
        validate_2d_array(X)
        geom_points = gp.GeoDataFrame(crs=self.projection,
                                      geometry=[geometry.Point(xi) for xi in X])
        ix = gp.tools.sjoin(geom_points, self.region, how='left')['index_right']
        ix[np.isnan(ix)] = -1

        return np.array(ix).astype(int)


    def _get_geometry_grid(self, size=40):

        return np.meshgrid(np.linspace(*self.box['x'], size), np.linspace(*self.box['y'], size))


    def points_to_frame(self, X):
        ## Number of points per geometry

        ix = self.locate(X)
        z = pd.DataFrame(data={'tile': ix[ix > -1], 'var_0': np.ones(sum(ix > -1))})
        z.dropna(inplace=True)
        z.groupby(by='tile').sum()
        z.index.name = None

        return z


    def marked_points_to_frame(self, X, Y, fun='mean'):

        # Check Y dimensions
        if Y.ndim == 1:
            Y = Y[:, None]
        assert Y.shape[0] == X.shape[0], 'X and Y dimensions do not match.'

        # Define dataframe of variables and tiles
        ix = self.locate(X)
        mask_ = ix > -1
        dict_ = {'var_%s' %j: Y[mask_, j] for j in np.arange(Y.shape[1])}
        dict_.update({'tile': ix[mask_]})
        z = pd.DataFrame(dict_)
        z.dropna(inplace=True)

        # Group dataframe
        if fun == 'sum':
            z = z.groupby(by='tile').sum()
        elif fun == 'mean':
            z = z.groupby(by='tile').mean()
        else:
            raise NotImplementedError

        z.index.name = None

        return z


    def raster_to_frame(self, raster, thresholds=[None, None], fun='mean', fill_missing=False):

        rr = RasterImage(image=raster, thresholds=thresholds)
        X = rr.get_coordinates(filter=True)
        y = rr.region.ReadAsArray().flatten()

        # Apply thresholds
        if thresholds[0] is not None:
            y = y[y >= thresholds[0]]
        if thresholds[1] is not None:
            y = y[y <= thresholds[0]]

        z = self.marked_points_to_frame(X, y, fun=fun)

        if fill_missing: # TODO: improve fill__missing functionality
            a = np.array(z.index)
            b = np.delete(np.array(self.region.index), a)
            if b.size > 0:
                Xa = X[a]
                Xb = X[b]
                Xa = np.hstack([Xa, (Xa[:, 0] * Xa[:, 1])[:, None]])
                Xb = np.hstack([Xb, (Xb[:, 0] * Xb[:, 1])[:, None]])
                ya = z['var_0']
                m = DecisionTreeRegressor()
                m.fit(Xa, ya)
                yb = m.predict(Xb)
                z = pd.DataFrame(data={'var_0': np.hstack([ya, yb])}, index=np.hstack([a, b]))
                z.sort_index(inplace=True)

        return z


    def plot(self, ax=None, color='gray', aspect='equal'):

        if ax is None:
            ax = self._get_canvas(aspect=aspect)

        for gi in self.region.geometry:
            ax.add_patch(PolygonPatch(gi, color=color, alpha=.5))

        return ax

    def _get_canvas(self, aspect='equal'):
        ax = plt.subplot()
        ax.set_xlim(self.box['x'])
        ax.set_ylim(self.box['y'])
        ax.set_aspect(aspect)
        return ax


