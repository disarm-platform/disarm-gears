import numpy as np
from ..validators import *


def binomial_to_bernoulli(n_positive, n_trials, X=None, aggregated=True):
    '''
    Takes a set binomial observations and returns a set of weighted bernoulli observations.

    :param n_positive: Number of successes in a Binomial experiment.
              Numpy array, shape = [n, ]

    :param n_trials: Number of trials in a Binomial experiment.
                     Numpy array, shape = [n, ]

    :param X: Features associated to n_positive (optional).
              Numpy array, shape = [n, ] or [n, d] (default None).

    :param aggregated: Whether to aggregate repeated observations.
                       Bool (default True).

    :return: tuple (ones-zeros array [m, ], weights array [m, ], new_X array [m, ] or [m, d])
    '''

    # Validate arrays
    validate_1d_array(n_positive, size=None)
    validate_1d_array(n_trials, size=n_positive.size)
    validate_integer_array(n_positive)
    validate_integer_array(n_trials)
    validate_non_negative_array(n_positive)
    validate_positive_array(n_trials)

    positive_ix = n_positive > 0
    negative_ix = n_trials > n_positive

    weights = np.hstack([n_positive[positive_ix], (n_trials-n_positive)[negative_ix]]).astype(float)
    target = np.hstack([np.ones(sum(positive_ix)), np.zeros(sum(negative_ix))])
    new_X = None

    if X is not None:
        if X.ndim == 1:
            assert X.size == n_positive.size, 'n_positive and X sizes do not match.'
            new_X = np.hstack([X[positive_ix], X[negative_ix]])
        elif X.ndim == 2:
            assert X.shape[0] == n_positive.size, 'n_positive size and X dimensions do not match.'
            new_X = np.vstack([X[positive_ix], X[negative_ix]])
        else:
            raise ValueError('X dimensions were not understood.')

    if not aggregated:
        # Repeat 0-1 as many times as n_positive-n_trials and add weight one to each of them.

        if X is not None:
            new_X = np.vstack([np.repeat(_x, _w).reshape(new_X.shape[1], -1).T for
                               _x,_w in zip(new_X, weights)])

        target = np.hstack([_y * np.ones(_w) for _y,_w in zip(target, weights.astype(int))])
        weights = np.ones_like(target)

    return target, weights, new_X

