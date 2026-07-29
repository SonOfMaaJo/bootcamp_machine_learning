import numpy as np


def loss_(y, y_hat):
    """Computes the half mean-squared-error of two non-empty numpy.arrays,
    without any for loop.
        The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a one-dimensional array of size m.
        y_hat: has to be an numpy.array, a one-dimensional array of size m.
    Returns:
        The half mean-squared-error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape or len(y.shape) != 1 or len(y_hat.shape) != 1:
        return None
    diff = y_hat - y
    return np.sum(diff * diff) / (2 * diff.size)
