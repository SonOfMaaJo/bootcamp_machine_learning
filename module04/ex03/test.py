import numpy as np
from logistic_loss_reg import reg_log_loss_


y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
print(reg_log_loss_(y, y_hat, theta, .5))
print(reg_log_loss_(y, y_hat, theta, .05))
print(reg_log_loss_(y, y_hat, theta, .9))
