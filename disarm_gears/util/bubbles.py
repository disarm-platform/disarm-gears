import numpy as np
from disarm_gears.validators import *

def bubbles(X, radius=.001, n_points=16):

    validate_2d_array(X, n_cols=2)
    assert radius >= 0, 'buffer radious must be non-negative.'
    assert isinstance(n_points, int)

    circle = np.pi * np.linspace(0, 2, n_points)#[1:]
    b = [np.vstack([xi + radius * np.array([np.cos(t), np.sin(t)]) for t in circle]) for xi in X]

    return b
