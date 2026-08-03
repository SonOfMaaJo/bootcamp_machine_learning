from math import floor
import numpy as np


def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y)
    into a training and a test set,
    while respecting the given proportion of examples to
    be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of
        the dataset that will be assigned to the
        training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible dimensions.
        None if x, y or proportion is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        return None
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.size:
        return None
    matrix = np.hstack((x, y))
    np.random.shuffle(matrix)
    nb_test = floor(proportion * x.shape[0])
    return (matrix[:nb_test, :x.shape[1]], matrix[nb_test:, :x.shape[1]],
            matrix[:nb_test, -1], matrix[nb_test:, -1])
