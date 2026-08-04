import numpy as np
from sigmoid import sigmoid_
from tools import add_intercept


def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from
    two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.ndim == 2 and theta.ndim == 2 and x.shape[1] == theta.shape[0] - 1:
        return sigmoid_(add_intercept(x) @ theta)
    return None


if __name__ == '__main__':
    x_ = np.array([4]).reshape((-1, 1))
    theta_ = np.array([[2], [0.5]])
    print(logistic_predict_(x_, theta_))
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(logistic_predict_(x2, theta2))
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(logistic_predict_(x3, theta3))
