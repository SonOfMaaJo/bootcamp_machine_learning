import numpy as np
from gradient import gradient
from loading import ft_progress


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
        (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of
        iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of dimension
        (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
        Raises:
        This function should not raise any Exception.
    """
    if not isinstance(alpha, (float, int)) or not isinstance(max_iter, int):
        return None
    grad = gradient(x, y, theta)
    if grad is None:
        return None
    print("fitting...")
    for _ in ft_progress(range(max_iter)):
        theta = theta - alpha * grad
        grad = gradient(x, y, theta)
    print("done.")
    return theta
