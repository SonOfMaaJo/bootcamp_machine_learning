import numpy as np
from l2_reg import l2


def reg_log_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a logistic regression model
    from two non-empty numpy.ndarray,
    without any for loop. The two arrays must have the same shapes.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta is empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or\
            not isinstance(theta, np.ndarray) or\
            not isinstance(lambda_, float):
        return None
    if y.ndim == 2 and y.shape[1] == 1 and y.shape == y_hat.shape and\
            theta.ndim == 2 and theta.shape[1] == 1:
        return - np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) +\
            (lambda_ / (2 * y.shape[0])) * l2(theta)
