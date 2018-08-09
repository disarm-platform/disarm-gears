import numpy as np


def binomial_to_bernoulli(y, n_trials, X=None, aggregated=True):
    '''
    Takes a set binomial observations and returns a set of bernoulli observations weighted by
    the number of trials.

    :param y: Number of successes in a Binomial experiment.
              Numpy array, shape = [n, ]

    :param n_trials: Number of trials in a Binomial experiment.
                     Numpy array, shape = [n, ]

    :param X: Covariates or features associated to y (optional).
              Numpy array, shape = [n, ] or [n, d] (defaults to None).

    :return: tuple (ones-zeros array [m, ], weights array [m, ], new_X array [m, ] or [m, d])
    '''


    assert y.ndim == 1, 'y is expected to be one-dimensional.'
    assert n_trials.ndim == 1, 'n_trials is expected to be one-dimensional.'
    assert y.size == n_trials.size, 'y and n_trials sizes do not match.'
    assert (np.round(y) - y).sum(0) == 0, 'y is expected to be an array of integers.'
    assert (np.round(n_trials) - n_trials).sum(0) == 0, 'n_trials is expected to be an array of integers.'
    assert sum(y < 0) == 0, 'y is expected to be non-negative.'
    assert sum(n_trials <= 0) == 0, 'n_trials is expected to be positive.'

    positive_ix = y > 0
    negative_ix = n_trials > y

    weights = np.hstack([y[positive_ix], (n_trials-y)[negative_ix]]).astype(float)
    new_y = np.hstack([np.ones(sum(positive_ix)), np.zeros(sum(negative_ix))])
    new_X = None

    if X is not None:
        if X.ndim == 1:
            assert X.size == y.size, 'y and X sizes do not match.'
            new_X = np.hstack([X[positive_ix], X[negative_ix]])
        elif X.ndim == 2:
            assert X.shape[0] == y.size, 'y size and X dimensions do not match.'
            new_X = np.vstack([X[positive_ix], X[negative_ix]])
        else:
            raise ValueError('X dimensions were not understood.')

    if not aggregated: #TODO this is not tested yet

        if X is not None:
            new_X = np.vstack([np.repeat(_x, _w).reshape(new_X.shape[1], -1).T for
                               _x,_w in zip(new_X, weights)])
        new_y = np.hstack([_y * np.ones(_w) for _y,_w in zip(new_y, weights.astype(int))])
        weights = np.ones_like(new_y)

    return new_y, weights, new_X

