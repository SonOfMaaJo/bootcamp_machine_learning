import matplotlib.pyplot as plt
from prediction import predict_
import numpy as np


def plot(x, y, theta):
    """Plot the data and prediction line from tree non-empty numpy.array.
    Args:
    _____
        x: has to be an numpy.array, a one-dimensional array of size m.
        y: has to be an numpy.array, a one-dimensional array of size m.
        theta: has to be an numpy.array, a two-dimensional array of shape
        2 * 1.
    Returns:
    _____
        Nothing.
    Raises:
        This function should not rais any Exceptions.
    """
    y_hat = predict_(x, theta)
    if y_hat is None:
        pass
    plt.plot(x, y_hat, 'r-')
    plt.plot(x, y, 'bo')
    plt.show()
