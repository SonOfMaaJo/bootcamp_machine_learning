import matplotlib.pyplot as plt
from prediction import predict_
from vec_loss import loss_
import numpy as np


def plot_with_loss(x, y, theta):
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
    cost = loss_(y, y_hat)
    if cost is None:
        pass
    plt.plot(x, y_hat, 'r-')
    plt.plot(x, y, 'bo')
    y_starts = np.minimum(y, y_hat)
    y_ends = np.maximum(y, y_hat)
    plt.vlines(
        x=x,
        ymin=y_starts,
        ymax=y_ends,
        colors='red',
        linestyles='dashed',
        linewidth=2
    )
    plt.title(f"Cost: {cost}")
    plt.show()
