import numpy as np
import pandas as pd
import geopandas
from shapely import geometry
from disarm_gears.validators import validate_2d_array, validate_1d_array


class PointPattern:

    def __init__(self, points, attributes=None, crs=None):
        '''
        General class to handle point pattern observations in space.

        :param points: Set of coordinates.
                       Numpy array, shape = [n_points, 2].
        :param attributes: Attributes associated to the points (optional).
                           Numpy array or pandas DataFrame, shape = [n_points, m].
        :param crs: Coordinate reference system (optional).
                    String.
        '''
        # NOTE: The use of crs is not tested yet!!!

        # Validate input points
        if isinstance(points, pd.DataFrame):
            points = np.array(points)
        validate_2d_array(points, n_cols=2)

        if crs is not None:
            #assert isinstance(crs, str)
            raise NotImplementedError

        # Define points geometry
        _region = {'geometry': [geometry.Point(pi) for pi in points]}
        if attributes is not None:
            _region = self._add_attributes(region_dict=_region, attributes=attributes)

        # Define region as GeoDataFrame
        self.centroids = points
        self.projection = crs
        self.region = geopandas.GeoDataFrame(_region, crs=self.projection)
        self.region.index.name = None

        # Bounding box
        _box = np.asarray(geometry.MultiPoint(points).bounds)
        self.box = pd.DataFrame({'x': _box[[0, 2]], 'y': _box[[1, 3]]})


    def _add_attributes(self, region_dict, attributes):
        '''Add attributes to the region.'''
        #n_points = region_dict['geometry'].size
        n_points = len(region_dict['geometry'])
        if isinstance(attributes, pd.DataFrame):
            assert attributes.shape[0] == n_points, 'Attributes and points dimensions do not match.'
            region_dict.update({'%s' %ai: attributes[ai] for ai in attributes.columns})
            self.attributes_names = ['%s' %ai for ai in attributes.columns]
        elif isinstance(attributes, np.ndarray):
            if attributes.ndim == 1:
                assert attributes.size == n_points, 'Attributes and points dimensions do not match.'
                n_attributes = 1
                region_dict.update({'var_0': attributes})
                self.attributes_names = ['var_%s' %i for i in range(n_attributes)]
            elif attributes.ndim == 2:
                assert attributes.shape[0] == n_points, 'Attributes and points dimensions do not match.'
                n_attributes = attributes.shape[1]
                region_dict.update({'var_%s' %i: ai for i, ai in enumerate(attributes.T)})
                self.attributes_names = ['var_%s' %i for i in range(n_attributes)]
            else:
                raise ValueError('attributes dimensions not understood.')
        else:
            raise ValueError('attributes type not understood.')

        return region_dict


    def attributes_array(self):
        '''
        Return an array of attributes associated to the region, if any.

        :return: Array of attributes.
                 Numpy array or None(if there are no attributes).

        '''
        if not hasattr(self, 'attributes_names'):
            _attr = None
        else:
            _attr = np.array(self.region.loc[:, self.attributes_names])
        return _attr


    def set_boundary(self, B):
        '''
        Set a boundary to delimit the spatial region.

        :param B: Polygon(s) that define the region.
                  Geopandas DataFrame.
        '''
        assert isinstance(B, geopandas.GeoDataFrame)
        ix = np.repeat(False, self.region.shape[0])
        for i, pi in enumerate(self.region.geometry):
            for bj in B.geometry:
                if bj.contains(pi):
                    ix[i] = True
                    break

        self.region = self.region[ix]
        self.boundary = B

        # Re-define bounding box
        self.box.loc[0, 'x'] = B.bounds.minx[0]
        self.box.loc[1, 'x'] = B.bounds.maxx[0]
        self.box.loc[0, 'y'] = B.bounds.miny[0]
        self.box.loc[1, 'y'] = B.bounds.maxy[0]


    def make_grid(self, resolution=(10, 10), bounded=True):
        '''
        Return a regular grid of points across the region.

        :param resolution: Number of horizontal and vertical cells in the squared grid.
                           Tuple of integers (n, m).
        :param bounded: Whether to return all cells in a squared grid or just the cells within the boundary.
                        Boolean (default True)
        :return: Grid.
                 Numpy array, shape [n, 2]
        '''
        x = np.linspace(*self.box.x, resolution[0])
        y = np.linspace(*self.box.y, resolution[1])
        G = np.vstack([mi.ravel() for mi in np.meshgrid(x, y)]).T

        if bounded and hasattr(self, 'boundary'):
            ix = np.hstack([self.boundary.contains(geometry.Point(gi)) for gi in G])
            G = G[ix]

        return G


    def make_attributes_series(self, knots, var_name='knot'):
        '''Make a (time) series of attributes, by repeating them along the knots.'''
        assert isinstance(var_name, str), 'var_name must be a string object.'
        validate_1d_array(knots)
        knots = np.unique(knots)
        num_knots = knots.size
        a_series = pd.concat([self.region.loc[:, self.attributes_names]] * num_knots)
        a_series[var_name] = np.hstack([np.repeat(ki, self.region.shape[0]) for ki in knots])

        return a_series


    #TODO def raster_to_frame(self, raster, buffer, fun='mean', fill_method=None):
    '''
    from descartes import PolygonPatch
    import matplotlib.pyplot as plt
    
    def plot_boundary(self, ax=None, color='gray', aspect='equal'):

        if ax is None:
            ax = self._get_canvas(aspect=aspect)

        for gi in self.boundary.geometry:
            ax.add_patch(PolygonPatch(gi, color=color, alpha=.5))

        return ax


    def plot(self, ax=None, color='black', aspect='equal'):

        if ax is None:
            ax = self._get_canvas(aspect=aspect)

        if hasattr(self, 'boundary'):
            self.plot_boundary(ax=ax)

        ax.plot(*self.centroids.T, 'o', color=color)

        return ax

    def _get_canvas(self, aspect='equal'):
        ax = plt.subplot()
        ax.set_xlim(self.box['x'])
        ax.set_ylim(self.box['y'])
        ax.set_aspect(aspect)
        return ax
    '''
