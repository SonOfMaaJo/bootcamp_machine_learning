import numpy as np
from tools import add_intercept


def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of
    three non-empty numpy.ndarray,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1,
        containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray) or\
            not isinstance(theta, np.ndarray) or\
            not isinstance(lambda_, (float, int)):
        return None
    if x.ndim == 2 and y.ndim == 2 and x.shape[0] == y.shape[0] and\
            theta.ndim == 2 and theta.shape[0] == x.shape[1] + 1 and\
            theta.shape[1] == 1:
        X = add_intercept(x)
        grad = np.zeros((theta.shape))
        grad[0, 0] = np.mean(X @ theta - y)
        for j in range(1, theta.shape[0]):
            grad[j, 0] = (1 / x.shape[0]) * (
                np.sum((X @ theta - y) * X[:, j].reshape(-1, 1)) + lambda_ *
                theta[j, 0])
        return grad
    return None


def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of
    three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1,
        containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray) or\
            not isinstance(theta, np.ndarray) or\
            not isinstance(lambda_, (float, int)):
        return None
    if x.ndim == 2 and y.ndim == 2 and x.shape[0] == y.shape[0] and\
            theta.ndim == 2 and theta.shape[0] == x.shape[1] + 1 and\
            theta.shape[1] == 1:
        X = add_intercept(x)
        return (1 / x.shape[0]) * (np.transpose(X) @ (X @ theta - y) +
                                   lambda_ * np.vstack(
                                       (np.array([[0]]), theta[1:, :])))
    return None
