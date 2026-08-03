import numpy as np
from l2_reg import l2


def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model
    from two non-empty numpy.array,
    without any for loop. The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta are empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) \
            or not isinstance(theta, np.ndarray) or not isinstance(lambda_,
                                                                   float):
        return None
    if y.ndim == 2 and y.shape[1] == 1 and y.shape == y_hat.shape and \
            theta.ndim == 2 and theta.shape[1] == 1:
        return 0.5 / y.shape[0] * (np.sum((y_hat - y) ** 2) +
                                   lambda_ * l2(theta))
    return None
