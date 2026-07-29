import numpy as np
from math import sqrt


def mse_(y, y_hat):
    """Description:
            Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be a numpy.array, a two-dimensional vector of shape m * 1.
    Returns:
        mse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape[0] != y_hat.shape[0] or y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    diff = y_hat - y
    return np.sum(diff * diff) / diff.size


def rmse_(y, y_hat):
    """Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    mse = mse_(y, y_hat)
    if mse is None:
        return None
    return sqrt(mse)


def mae_(y, y_hat):
    """Description:
            Calculate the MAE between the predicted output and the real output.
        Args:
        y: has to be a numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    mae: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape[0] != y_hat.shape[0] or y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    diff = y_hat - y
    return np.sum(abs(diff)) / diff.size


def r2score_(y, y_hat):
    """Description:
        Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape[0] != y_hat.shape[0] or y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    diff_m = y - np.mean(y)
    return 1 - diff_m.size * mse_(y, y_hat) / np.sum(diff_m * diff_m)
