import numpy as np
from mylinearregression import MyLinearRegression


class MyRidge(MyLinearRegression):
    """
    Description:
        My personnal ridge regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        super.__init__(theta, alpha, max_iter)
        if not isinstance(lambda_, (int, float)):
            raise TypeError("wrong type for lambda_")
        self.lambda_ = lambda

    def set_params_(self, thetas, alpha, max_iter, lambda_):
        self.thetas = thetas
        self.alpha = alpha
        self.max_iter = max_iter
        self.lambda_ = lambda_

    def get_params_(self):
        return (self.thetas, self.alpha, self.max_iter, self.lambda_)

    def loss_(self, y, y_hat):
        loss = super().loss_(y, y_hat)
        return loss + self.lambda_ / (2 * y.shape[0]) * l2(self.theta)

    def gradient_(self, y, x):
        if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray) or\
            not isinstance(theta, np.ndarray) or\
            not isinstance(lambda_, (float, int)):
        return None
        if x.ndim == 2 and y.ndim == 2 and x.shape[0] == y.shape[0] and\
                theta.ndim == 2 and self.theta.shape[0] == x.shape[1] + 1:
            X = add_intercept(x)
            return (1 / x.shape[0]) * (np.transpose(X) @ (X @ self.theta - y) +
                                       self.lambda_ * np.vstack(
                                           (np.array([[0]]),
                                            self.theta[1:, :])))
        return None

    def fit_(self, x, y):
        grad = self.gradient_(x, y, self.thetas)
        if grad is None:
            return
        print(f"fitting{self.name} model...")
        for _ in ft_progress(range(self.max_iter)):
            self.thetas = self.thetas - self.alpha * grad
            grad = self.gradient_(x, y, self.thetas)
            self.max_iter -= 1
        print()
        print("done.")


