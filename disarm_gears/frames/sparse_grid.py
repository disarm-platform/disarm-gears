import numpy as np
import geopandas
import shapely


class SparseGrid:

    def __init__(self, x_lim, y_lim, n_cols=10, n_rows=10, tag_prefix = ''):
        '''
        General class to define a spatial frame composed of regular polygons,
        based on a grid of size n_cols x n_rows

        :param x_lim: Minimum and Maximum values in the horizontal axis.
                      Tupple of floats.
        :param y_lim: Minimum and Maximum values in the vertical axis.
                      Tupple of floats.
        :param n_cols: Number of columns in which the horizontal axis is divided.
                       Integer.
        :param n_rows: Number of columns in which the vertical axis is divided.
                       Integer
        :param tag_prefix: Prefix to use as id of the polygons in the grid.
                           String.
        '''

        assert len(x_lim) == 2 and np.diff(x_lim) > 0
        assert len(y_lim) == 2 and np.diff(y_lim) > 0
        assert isinstance(n_cols, int) and n_cols > 0
        assert isinstance(n_rows, int) and n_cols > 0
        assert isinstance(tag_prefix, str)
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.dx = (x_lim[1] - x_lim[0]) / n_cols
        self.dy = (y_lim[1] - y_lim[0]) / n_rows
        self.x_grid = np.linspace(x_lim[0], x_lim[1] - self.dx, n_cols)
        self.y_grid = np.linspace(y_lim[0], y_lim[1] - self.dy, n_rows)
        self.n_cols = n_cols
        self.n_rows = n_rows
        n_cells = self.n_cols * self.n_rows
        id_size = len(str(n_cells - 1))
        self.tag_prefix = tag_prefix
        self.tags = [self.tag_prefix + '0' * (id_size - len(str(f'{i}'))) + f'{i}' for i in range(n_cols * n_rows) ]
        self.sparse_frame = geopandas.GeoDataFrame({'id' :[], 'geometry' :None})

    def get_row(self, y):
        '''
        Get the row in the grid to which a value y corresponds

        :param y: Coordinate in the vertical axis
                  Float

        :return: Row number
                 Integer
        '''
        if y >= self.y_lim[0] or y <= self.y_lim[1]:
            return sum(self.y_grid <= y) - 1

    def get_col(self, x):
        '''
        Get the column in the grid to which a value x corresponds

        :param x: Coordinate in the horizontal axis
                  Float

        :return: Column number
                 Integer
        '''
        if x >= self.x_lim[0] or x <= self.x_lim[1]:
            return sum(self.x_grid <= x) - 1

    def tag_from_ij(self, i, j):
        '''
        Get the tag (or id) of a polygon based on its location within the grid

        :param i: Column number within the grid
                  Integer
        :param j: Row number within the grid
                  Integer

        :return: Tag
                 String
        '''
        ij = str(j * self.n_cols + i)
        return self.tag_prefix + '0' * (len(str(self.n_cols * self.n_rows)) - len(ij)) + ij

    def tag_from_xy(self, x, y):
        '''
        Get the tag (or id) of a polygon based on a pair of coordinates located within it

        :param x: Coordinate in the horizontal axis
                  Float
        :param y: Coordinate in the vertical axis
                  Float

        :return: Tag
                 String
        '''
        nx = self.get_col(x)
        ny = self.get_row(y)
        if nx is not None and ny is not None:
            return self.tag_from_ij(nx, ny)

    def ij_from_tag(self, tag):
        '''
        Get the location of a polygon within the grid based on its tag (or id)

        :param tag: id of a polygon
                    String

        :return: Location (i, j) of a polygon
                 Tuple of integers
        '''
        ix = self.tags.index(tag)
        ny = ix // self.n_cols
        nx = ix % self.n_cols
        return nx, ny

    def add_polygon_from_tag(self, tag):
        '''
        Incorporate a polygon to the sparse_grid GeoDataFrame

        :param tag: id of a polygon
                    String
        '''
        if tag not in self.sparse_frame.id.tolist():
            nx, ny = self.ij_from_tag(tag)
            x0 = self.x_lim[0] + nx * self.dx
            y0 = self.y_lim[0] + ny * self.dy
            sq = [(x0, y0), (x0, y0 + self.dy), (x0 + self.dx, y0 + self.dy), (x0 + self.dx, y0)]
            ngeo = geopandas.GeoDataFrame({'id': [tag],
                                           'geometry':  shapely.geometry.Polygon(sq)})
            self.sparse_frame = self.sparse_frame.append(ngeo)
        self.sparse_frame.reset_index(inplace=True, drop=True)

    def add_polygon_from_xy(self, X):
        '''
        Incorporate a polygon to the sparse_grid GeoDataFrame

        :param X: Points withing the grid
                  Numpy array of dimensions (n, 2)
        '''
        assert isinstance(X, np.ndarray)
        assert X.shape[1] == 2
        for xi in X:
            tagi = self.tag_from_xy(*xi)
            self.add_polygon_from_tag(tagi)

    def get_simplified(self, tolerance=1e-4):
        '''
        Simplify adjacent polygons in sparse_grid

        :param tolerance: Points in a simplified geometry will be no more than `tolerance` distance from the original.
                          (see geopandas.GeoDataFrame.simplify).
                          float

        :return: Simplified polygons object.
                 GeoDataFrame
        '''
        assert tolerance > 0
        mpolyg = shapely.geometry.multipolygon.asMultiPolygon(self.sparse_frame.geometry)
        mpolyg = mpolyg.simplify(tolerance=tolerance, preserve_topology=False)
        return geopandas.GeoDataFrame({'id': list(range(len(mpolyg))), 'geometry':  mpolyg})