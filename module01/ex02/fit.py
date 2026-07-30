from vec_gradient import simple_gradient


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1:
        (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1:
        (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during
        the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(alpha, (float, int)) or not isinstance(max_iter, int):
        return None
    grad = simple_gradient(x, y, theta)
    if grad is None:
        return None
    while max_iter:
        theta = theta - alpha * grad
        grad = simple_gradient(x, y, theta)
        max_iter -= 1
    return theta
