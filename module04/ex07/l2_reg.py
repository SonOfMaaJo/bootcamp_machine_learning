import numpy as np


def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray,
    with a for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(theta, np.ndarray):
        return None
    if theta.ndim == 2 and theta.shape[1]:
        result = 0
        for j in range(1, theta.shape[0]):
            result += theta[j, 0] ** 2
        return result
    return None


def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray,
    without any for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(theta, np.ndarray):
        return None
    return np.sum(theta[1:, 0] ** 2) if theta.ndim == 2 \
        and theta.shape[1] == 1 else None
