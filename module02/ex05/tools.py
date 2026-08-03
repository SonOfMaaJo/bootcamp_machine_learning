import numpy as np

def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
    _____
        x: has to be a numpy.array. x can be one-dimensional (m * 1) or
    two-dimensional (m * n) arry.
    Returns:
    _____
        X, a numpy.array of dimension m * (n + 1)
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception
    """
    if not isinstance(x, np.ndarray):
        return None
    x_dim = x.shape[0]
    if x_dim == 0:
        return None
    try:
        y_dim = 2 if len(x.shape) == 1 else x.shape[1] + 1
        X = np.ones((x_dim, y_dim))
        X[:, 1:] = x.reshape(-1, y_dim - 1)
        return X
    except Exception:
        pass
