import numpy as np


def simple_predict(x, theta):
    """computes the vector of predicition y_hat from two non-empty
    numpy.ndarray.
    Args:
    _____
        x:  has to be an numpy.ndarray, a one-dimensional array of size m.
        theta:  has to be an numpy.ndarray, a one-dimensional array of size 2.
    Returns:
    _____
        y_hat as a numpy.ndarray, a one-dimensional array of size m.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if (len(x.shape) != 1 or x.shape[0] == 0) \
            and (len(theta.shape) != 1 or theta.shape[0] != 2):
        return None
    try:
        y_hat = theta[1] * x + theta[0]
        return y_hat
    except Exception:
        pass
