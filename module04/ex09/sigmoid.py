import numpy as np


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.ndim == 2 and x.shape[1] == 1:
        x = np.asarray(x, dtype=float)
        result = np.empty_like(x)
        pos = x >= 0
        neg = ~pos
        result[pos] = 1 / (1 + np.exp(-x[pos]))
        result[neg] = np.exp(x[neg]) / (1 + np.exp(x[neg]))
        return result
    return None


if __name__ == '__main__':
    x = np.array([[-4]])
    print(sigmoid_(x))
    x = np.array([[2]])
    print(sigmoid_(x))
    x = np.array([[-4], [2], [0]])
    print(sigmoid_(x))
