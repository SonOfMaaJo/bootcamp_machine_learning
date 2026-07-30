import numpy as np
from tools import add_intercept


def predict_(x, theta):
    """
    Computes the Vector of prediction y_hat from two non-empty numpy.array.
    Args:
    _____
        x:  hat to be an numpy.array, a two-dimensional array of size m * 1.
        theta: has to be numpy.array, a two-dimensional array of shape 2 * 1.
    Returns:
    _____
        y_hat as a numpy.array, a two-dimensional array of shape m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
    Raises:
    _____
        this functions shouls not raise any Exceptions.
    """
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        return None
    if not isinstance(theta, np.ndarray) \
            or theta.size != 2 or theta.shape[1] != 1:
        return None
    X = add_intercept(x)
    return X @ theta
