import numpy as np
import geopandas
import shapely


class SparseGrid:

    def __init__(self, x_lim, y_lim, n_cols=10, n_rows=10, tag_prefix = ''):

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
        if y >= self.y_lim[0] or y <= self.y_lim[1]:
            return sum(self.y_grid <= y) - 1

    def get_col(self, x):
        if x >= self.x_lim[0] or x <= self.x_lim[1]:
            return sum(self.x_grid <= x) - 1

    def tag_from_ij(self, i, j):
        ij = str(j * self.n_cols + i)
        return self.tag_prefix + '0' * (len(str(self.n_cols * self.n_rows)) - len(ij)) + ij

    def tag_from_xy(self, x, y):
        nx = self.get_col(x)
        ny = self.get_row(y)
        if nx is not None and ny is not None:
            return self.tag_from_ij(nx, ny)

    def ij_from_tag(self, tag):
        ix = self.tags.index(tag)
        ny = ix // self.n_cols
        nx = ix % self.n_cols
        return nx, ny

    def add_polygon_from_tag(self, tag):
        if tag not in self.sparse_frame.id.tolist():
            nx, ny = self.ij_from_tag(tag)
            x0 = self.x_lim[0] + nx * self.dx
            y0 = self.y_lim[0] + ny * self.dy
            sq = [(x0, y0), (x0, y0 + self.dy), (x0 + self.dx, y0 + self.dy), (x0 + self.dx, y0)]
            ngeo = geopandas.GeoDataFrame({'id': [tag],
                                           'geometry':  shapely.geometry.Polygon(sq)})
            self.sparse_frame = self.sparse_frame.append(ngeo)

    def add_polygon_from_xy(self, X):
        for xi in X:
            tagi = self.tag_from_xy(*xi)
            self.add_polygon_from_tag(tagi)

    def get_simplified(self, tolerance=1e-4):
        mpolyg = shapely.geometry.multipolygon.asMultiPolygon(self.sparse_frame.geometry)
        mpolyg = mpolyg.simplify(tolerance=tolerance, preserve_topology=False)
        return geopandas.GeoDataFrame({'id': list(range(len(mpolyg))), 'geometry':  mpolyg})