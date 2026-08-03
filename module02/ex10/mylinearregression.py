from tools import add_intercept
from gradient import gradient
from loading import ft_progress
from math import sqrt
import numpy as np


class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000, name=''):
        if not isinstance(thetas, np.ndarray) or thetas.ndim != 2 or \
                thetas.shape[1] != 1:
            raise NotImplementedError("Invalid type of thetas.")
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.name = name

    def predict_(self, x):
        """
        Computes the Vector of prediction y_hat from two non-empty numpy.array.
         Args:
        _____
            x:  hat to be an numpy.array, a two-dimensional array of size
            m * 1.
        Returns:
        _____
            y_hat as a numpy.array, a two-dimensional array of shape m * 1.
            None if x are empty numpy.array.
            None if x  dimensions are not appropriate.
        Raises:
        _____
            this functions shouls not raise any Exceptions.
        """
        if not isinstance(x, np.ndarray) or x.ndim != 2 or \
                x.shape[1] != self.thetas.size - 1:
            return None
        return add_intercept(x) @ self.thetas

    def fit_(self, x, y):
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1:
            (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1:
            (number of training examples, 1).
        Returns:
            None if
        Raises:
            This function should not raise any Exception.
                """
        grad = gradient(x, y, self.thetas)
        if grad is None:
            return
        print(f"fitting{self.name} model...")
        for _ in ft_progress(range(self.max_iter)):
            self.thetas = self.thetas - self.alpha * grad
            grad = gradient(x, y, self.thetas)
            self.max_iter -= 1
        print()
        print("done.")

    def loss_elem_(self, y, y_hat):
        """
        Description:
            Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
            y: has to be an numpy.array, a two-dimensional array of shape
            m * 1.
            y_hat: has to be an numpy.array, a two-dimensional array of shape
            m * 1.
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
        if y.shape[0] != y_hat.shape[0] or y.shape[1] != 1 \
                or y_hat.shape[1] != 1:
            return None
        return (y_hat - y) ** 2

    def loss_(self, y, y_hat):
        """
        Description:
            Calculates the value of loss function.
        Args:
            y: has to be an numpy.array, a two-dimensional array of shape
            m * 1.
            y_hat: has to be an numpy.array, a two-dimensional array of
            shape m * 1.
        Returns:
            J_value : has to be a float.
            None if there is a dimension matching problem.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        J_elem = self.loss_elem_(y, y_hat)
        if J_elem is None:
            return None
        return np.sum(J_elem) / (2 * len(J_elem))

    def mse_(self, y, y_hat):
        """Description:
        Calculate the MSE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a two-dimensional array of shape m * 1.
            y_hat: has to be a numpy.array, a two-dimensional vector of
            shape m * 1.
        Returns:
            mse: has to be a float.
        None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.shape[0] != y_hat.shape[0] or y.shape[1] != 1 or \
                y_hat.shape[1] != 1:
            return None
        diff = y_hat - y
        return np.sum(diff * diff) / diff.size

    def rmse_(self, y, y_hat):
        """Description:
        Calculate the RMSE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a two-dimensional array of shape m * 1.
            y_hat: has to be a numpy.array, a two-dimensional array of shape
            m * 1.
        Returns:
            rmse: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
        """
        mse = self.mse_(y, y_hat)
        if mse is None:
            return None
        return sqrt(mse)

    def mae_(self, y, y_hat):
        """Description:
        Calculate the MAE between the predicted output and the real output.
            Args:
            y: has to be a numpy.array, a two-dimensional array of shape m * 1.
            y_hat: has to be a numpy.array, a two-dimensional array of
            shape m * 1.
        Returns:
        mae: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.shape[0] != y_hat.shape[0] or y.shape[1] != 1 or\
                y_hat.shape[1] != 1:
            return None
        diff = y_hat - y
        return np.sum(abs(diff)) / diff.size

    def r2score_(self, y, y_hat):
        """Description:
        Calculate the R2score between the predicted output and the output.
        Args:
            y: has to be a numpy.array, a two-dimensional array of shape m * 1.
            y_hat: has to be a numpy.array, a two-dimensional array of shape
            m * 1.
        Returns:
            r2score: has to be a float.
            None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exceptions.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.shape[0] != y_hat.shape[0] or y.shape[1] != 1 or\
                y_hat.shape[1] != 1:
            return None
        diff_m = y - np.mean(y)
        return 1 - diff_m.size * self.mse_(y, y_hat) / np.sum(diff_m * diff_m)
