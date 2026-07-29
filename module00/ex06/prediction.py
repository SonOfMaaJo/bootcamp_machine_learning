import numpy as np
from tools import add_intercept

def predict_(x, theta):
    """
    Computes the Vector of prediction y_hat from two non-empty numpy.array.
    Args:
    _____
        x:  hat to be an numpy.array, a one-dimensional array of size m.
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
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    if not isinstance(theta, np.ndarray) \
            or theta.size == 0 or theta.shape[0] != 2:
        print('yes')
        return None
    try:
        X = add_intercept(x)
        return (theta[0, 0] * X[:, 0] + theta[1, 0] * X[:, 1]).reshape(-1, 1)
    except Exception:
        pass
