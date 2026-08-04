import numpy as np
from tools import add_intercept
from sigmoid import sigmoid_
from vec_log_gradient import vec_log_gradient
from loading import ft_progress


class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression to classify things.
    """
    supported_penalties = ['l2']
    def __init__(self, theta, alpha=0.001, max_iter=1000, penality='l2',
                 eps=1e-15, lambda_=1.0):
        if not isinstance(alpha, float):
            raise TypeError("Wrong type for alpha (float)")
        if not isinstance(theta, np.ndarray):
            raise TypeError("Wrong type for theta (np.array of dim m * 1)")
        if not isinstance(max_iter, int):
            raise TypeError("Wrong type for max_iter (int)")
        if not isinstance(eps, float):
            raise TypeError("Wrong type for epsilon (float)")
        self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter
        self.eps = eps
        self.penality = penality
        self.lambda_ = lambda_ if penality in self.supported_penalties else 0

    def predict_(self, x):
        """Computes the vector of prediction y_hat from
        a non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a vector of dimension m * n.
        Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
            None if x is empty numpy.ndarray.
            None if x dimensions is not appropriate.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray):
            return None
        if x.ndim == 2 and x.shape[1] == self.theta.shape[0] - 1:
            return sigmoid_(add_intercept(x) @ self.theta)
        return None

    def loss_elem_(self, y, yhat):
        """
        Computes the vector of loss elements from two non-empty numpy.ndarray
        Args:
            y: has to be an numpy.ndarray, a vector of dimension m * 1
            y_hat: has to be an numpy.ndarry, a vector of dimension m * 1
        Returns:
            loss_el as a numpy.ndarry, a vector of dimension m * 1
            None if one of y, y_hat is empty numpy.ndarray
            None if y, y_hat dimensions are not appropriate.
        Raises:
            This function should not raise any Exception
        """
        if not isinstance(y, np.ndarray) or not isinstance(yhat, np.ndarray):
            return None
        if y.ndim == 2 and y.shape[1] == 1 and y.shape == yhat.shape:
            return -y * np.log(yhat + self.eps) -\
                (1 - y) * np.log(1 - yhat + self.eps)
        return None

    def loss_(self, y, yhat):
        """
        Computes the logistic loss value.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            eps: epsilon (default=1e-15)
        Returns:
            The logistic loss value as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(yhat, np.ndarray):
            return None
        if y.ndim == 2 and y.shape[1] == 1 and yhat.shape == y.shape:
            return np.mean(self.loss_elem_(y, yhat)) + self.lambda_ *\
                1 / (2 * y.size) * np.vstack((np.array([[0]]),
                                              self.theta[1:, :]))
        return None

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
        grad = vec_log_gradient(x, y, self.theta)
        if grad is None:
            return
        penal_vector = np.vstack((np.array([[0]]), self.theta[1:, :]))
        for _ in ft_progress(range(self.max_iter)):
            self.theta = self.theta - self.alpha * grad - 1 / y.size\
                * self.lambda_ * penal_vector
            grad = vec_log_gradient(x, y, self.theta)
        print()
        print("done.")
