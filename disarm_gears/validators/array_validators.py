import numpy as np


def validate_1d_array(x, size=None):
    '''Validate type and dimensions of an object x.'''

    assert isinstance(x, np.ndarray), 'Expecting a numpy array.'
    assert x.ndim == 1, 'Expecting a one-dimensional array.'

    if size is not None:
        assert x.size == size, 'Array size is different from expected.'


def validate_2d_array(x, n_cols=None, n_rows=None):
    '''Validate type and dimensions of an object x.'''
    assert isinstance(x, np.ndarray), 'Expecting a numpy array.'
    assert x.ndim == 2, 'Expecting a two-dimensional array.'

    if n_rows is not None:
        assert x.shape[0] == n_rows, 'Array size is different from expected.'

    if n_cols is not None:
        assert x.shape[1] == n_cols, 'Number of columns is different from expected.'


def validate_integer_array(x):
    '''Validate array elements are integers.'''
    assert (np.round(x) == x).all(), 'Expecting an array of integers.'


def validate_positive_array(x):
    '''Validate array elements are positive.'''
    assert (x > 0).all(), 'Expecting array of positive elements.'


def validate_non_negative_array(x):
    '''Validate array elements are non-negative.'''
    assert (x >= 0).all(), 'Expecting array of non-negative elements.'

