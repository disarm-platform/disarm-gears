import numpy as np
from disarm_gears.validators import validate_2d_array


def trend_1st_order(X):
    '''Bilinear surface trend.'''
    validate_2d_array(X, n_cols=2)
    return np.hstack([X, X.prod(axis=1)[:, None]])


def trend_2nd_order(X):
    '''2nd order surface trend.'''
    validate_2d_array(X, n_cols=2)
    return np.hstack([X**2., X, X.prod(axis=1)[:, None]])


def trend_3rd_order(X):
    '''3rd order surface trend.'''
    validate_2d_array(X, n_cols=2)
    _a2b = X[:, 0]**2. * X[:, 1]
    _ab2 = X[:, 0] * X[:, 1]**2.
    return np.hstack([X**3., X**2., X, _a2b[:, None], _ab2[:, None], X.prod(axis=1)[:, None]])

