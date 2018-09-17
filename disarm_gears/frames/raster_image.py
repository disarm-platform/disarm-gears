import numpy as np
import pandas as pd
from osgeo import gdal
from affine import Affine
from disarm_gears.validators import *

import matplotlib.pyplot as plt

class RasterImage:

    def __init__(self, image, thresholds=[None, None]):
        '''
        Define a frame based on a raster image.

        :param image: Raster image.
                      Object of type osgeo.gdal.Dataset.
        :param thresholds: Minimum and Maximum pixel values accepted (optional).
                           List of size 2: [min_val, max_val].
        '''
        if image.GetLayerCount() > 0:
            #TODO
            raise NotImplementedError

        assert isinstance(image, gdal.Dataset)
        self.region = image
        self.projection = self.region.GetProjection()
        assert isinstance(thresholds, list), 'thresholds must be a list.'
        assert len(thresholds) == 2, 'thresholds must be a list of length 2.'
        assert np.all([i is None or isinstance(i, float) or isinstance(i, int) for i in thresholds]),\
            'threshold elements must be None, float or integer.'
        if np.all([i is not None for i in thresholds]):
            assert thresholds[0] < thresholds[1], 'threshold values do not have increasing order.'
        self.thresholds = thresholds

        # Bounding box
        pix = np.array([[0, 0], [image.RasterYSize, image.RasterXSize]])
        X = self.coords_from_pixel(pix)
        self.box = pd.DataFrame({'x': [X[:, 0].min(), X[:, 0].max()],
                                 'y': [X[:, 1].min(), X[:, 1].max()]})


    def get_coordinates(self, filter=True):
        '''
        Get pixel coordinates of the image.

        :parameter filter: If pixels with values outside the threshold should be removed.
                           Bool object (default True).
        :return: Array of coordinates.
        '''
        lng_size = self.region.RasterYSize
        lat_size = self.region.RasterXSize

        pix_rows = np.repeat(np.arange(lng_size)[:, None], lat_size, axis=1)
        pix_cols = np.repeat(np.arange(lat_size)[None, :], lng_size, axis=0)

        if filter and not np.all([i is None for i in self.thresholds]):
            if np.all([i is not None for i in self.thresholds]):
                rarr = self.region.ReadAsArray()
                mask = np.logical_and(self.thresholds[0] <= rarr, rarr <= self.thresholds[1])
            elif self.thresholds[0] is not None:
                mask = self.thresholds[0] <= self.region.ReadAsArray()
            else:
                mask = self.region.ReadAsArray() <= self.thresholds[1]
            pix_rows = pix_rows[mask]
            pix_cols = pix_cols[mask]

        X = np.vstack([pix_rows.flatten(), pix_cols.flatten()]).T

        return self.coords_from_pixel(X)


    def pixel_from_coords(self, X):
        '''
        Get corresponding pixels of an array of coordinates.
        This is a wrapper around the affine Python library.

        :param X: Coordinates.
                  Numpy array, shape = [n, 2].
        :return: Array of pixels.
                 Numpy array, shape[n, 2].
        '''
        validate_2d_array(X)
        validate_non_negative_array(X)
        validate_integer_array(X)
        reverse_transform = ~Affine.from_gdal(*self.region.GetGeoTransform())
        col, row = reverse_transform * (X[:, 0], X[:, 1])
        col, row = col.astype(int), row.astype(int)

        return np.vstack([row, col]).T


    def coords_from_pixel(self, X):
        '''
        Get corresponding coordinates form an array of pixels.
        This is a wrapper around the affine Python library.

        :param X: Pixel locations.
                  Numpy array of non negative integers, shape = [n, 2]
        :return: Array of coordinates.
                 Numpy array, shape[n, 2].
        '''

        validate_2d_array(X)
        forward_transform = Affine.from_gdal(*self.region.GetGeoTransform())
        lng, lat = forward_transform * (X[:, 1], X[:, 0])

        return np.vstack([lng, lat]).T


    def extract_value(self, X):
        '''
        Get pixel values at specific coordinate locations.

        :param X: Coordinates.
                  Numpy array, shape = [n, 2].
        :return: Array of pixels.
                 Numpy array, shape[n, ].
        '''
        validate_2d_array(X)
        pix = self.pixel_from_coords(X)

        return self.region.ReadAsArray()[pix[:, 0], pix[:, 1]]


    def plot(self, ax=None, aspect='equal', sample_factor=2):
        '''Plot the region'''
        #TODO add test
        if ax is None:
            ax = self._get_canvas(aspect=aspect)

        # Note: lng are cols and lat are rows
        offset = int(sample_factor/2)
        pix_rows = np.arange(offset, self.region.RasterYSize - offset, step=sample_factor)
        pix_cols = np.arange(offset, self.region.RasterXSize - offset, step=sample_factor)
        _y = np.vstack([pix_rows, np.zeros_like(pix_rows)]).T
        _x = np.vstack([np.zeros_like(pix_cols), pix_cols]).T
        _lng = self.coords_from_pixel(_x)[:, 0]
        _lat = self.coords_from_pixel(_y)[:, 1]

        grid_lng, grid_lat = np.meshgrid(_lng, _lat)
        grid_cols, grid_rows = np.meshgrid(pix_cols, pix_rows)

        Z_band = self.region.GetRasterBand(1)
        Z_array = Z_band.ReadAsArray()
        Z = Z_array[grid_rows.ravel(), grid_cols.ravel()].reshape(grid_rows.shape)
        self.Z = Z
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.grid_lng = grid_lng
        self.grid_lat = grid_lat
        ax.contourf(grid_lng, grid_lat, Z)

        return ax

    def _get_canvas(self, aspect='equal'):
        #TODO add test
        ax = plt.subplot()
        ax.set_xlim(self.box['x'])
        ax.set_ylim(self.box['y'])
        ax.set_aspect(aspect)

        return ax
