import numpy as np
from math import sqrt


def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray
    using the z-score standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x’ as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn’t raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.ndim == 0 or (x.ndim == 2 and x.shape[0] != 1 and x.shape[1] != 1):
        return None
    mu = (1 / x.size) * np.sum(x)
    sigma = (1 / (x.size - 1)) * np.sum((x - mu) ** 2)
    return (x - mu) / sqrt(sigma)
