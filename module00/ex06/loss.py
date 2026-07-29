import numpy as np


def loss_elem_(y, y_hat):
    """
    Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
        y: has to be an numpy.array, a two-dimensional array of shape m * 1.
        y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
    Returns:
        J_elem: numpy.array, a array of dimension
            (number of the training examples, 1).
        None if there is a dimension matching problem.
        None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape[0] != y_hat.shape[0] or y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    try:
        return (y_hat - y) ** 2
    except Exception:
        pass

def loss_(y, y_hat):
    """
        Description:
            Calculates the value of loss function.
        Args:
            y: has to be an numpy.array, a two-dimensional array of shape m * 1.
            y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
        Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
    """
    J_elem = loss_elem_(y, y_hat)
    if J_elem is None:
        return None
    return np.sum(J_elem) / (2 * len(J_elem))
