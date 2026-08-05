import numpy as np
from mylinearregression import MyLinearRegression
from functools import wraps
from tools import add_intercept
from l2_reg import l2
from loading import ft_progress


def copy_doc(from_func):
    def decorator(to_func):
        to_func.__doc__ = from_func.__doc__
        return to_func
    return decorator


class MyRidge(MyLinearRegression):
    """
    Description:
        My personnal ridge regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        super().__init__(thetas, alpha, max_iter)
        if not isinstance(lambda_, (int, float)):
            raise TypeError("wrong type for lambda_")
        self.lambda_ = lambda_

    def set_params_(self, thetas=None, alpha=None, max_iter=None,
                    lambda_=None):
        """
        set the parameters of the models.
        Args:
            thetas: has to be a np.array of dimensions n * 1
            alpa:   has to be a float
            max_iter: has to be an integer
            lambda_: has to be an flot
        Returns:
            None
        Raises:
            This function should not raise any exception.
        """
        if thetas is not None:
            self.thetas = thetas
        if alpha is not None:
            self.alpha = alpha
        if max_iter is not None:
            self.max_iter = max_iter
        if lambda_ is not None:
            self.lambda_ = lambda_

    def get_params_(self):
        """
        get the paramaters of the function as a tuple
        (thetas, alpha, max_iter, lambda_)
        """
        return (self.thetas, self.alpha, self.max_iter, self.lambda_)

    @copy_doc(MyLinearRegression.loss_elem_)
    def loss_elem_(self, y, y_hat):
        loss_elem = super().loss_elem_(y, y_hat)
        return loss_elem + self.lambda_ * l2(self.thetas)

    @copy_doc(MyLinearRegression.loss_)
    def loss_(self, y, y_hat):
        loss_elem = self.loss_elem_(y, y_hat)
        return 0.5 * np.mean(loss_elem) if loss_elem is not None else None

    def gradient_(self, x, y):
        """
        Computes the regularized linear gradient of three non-empty
        numpy.ndarray,
        without any for-loop. the two arrays must have compatible shape.
        Args:
            y: has to be a numpy.ndarray, a vector of shape m * 1.
            x: has to be a numpy.ndarray, a matrix of dimension m * n
        Return:
            A numpy.ndarray, a vector of shape (n + 1) * 1, containing the
            results of the formula for all j.
            None if y, x are empty numpy.ndarray.
            None if y, x does not share compatibles shape.
            None if y, x is not of the expected type.
        Raises:
            This function should not raise any Exception
        """
        if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray):
            return None
        if x.ndim == 2 and y.ndim == 2 and x.shape[0] == y.shape[0] and\
                self.thetas.shape[0] == x.shape[1] + 1:
            X = add_intercept(x)
            return (1 / x.shape[0]) * (np.transpose(X) @ (X @ self.thetas - y)
                                       + self.lambda_ * np.vstack(
                                           (np.array([[0]]),
                                            self.thetas[1:, :])))
        return None

    @copy_doc(MyLinearRegression.fit_)
    def fit_(self, x, y):
        grad = self.gradient_(x, y)
        if grad is None:
            return
        for _ in ft_progress(range(self.max_iter)):
            self.thetas = self.thetas - self.alpha * grad
            grad = self.gradient_(x, y)
        print("done.")
