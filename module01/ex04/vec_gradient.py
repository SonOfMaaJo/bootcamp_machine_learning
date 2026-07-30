from tools import add_intercept
import numpy as np


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.arrays,
    with a for-loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not \
            isinstance(theta, np.ndarray):
        return None
    if x.ndim != 2 or x.shape[1] != 1 or x.shape != y.shape \
            or theta.ndim != 2 or theta.size != 2:
        return None
    X = add_intercept(x)
    return (1 / x.size) * np.transpose(X) @ (X @ theta - y)
