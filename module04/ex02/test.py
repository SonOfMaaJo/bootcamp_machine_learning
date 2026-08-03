import numpy as np
from linear_loss_reg import reg_loss_


y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
print(reg_loss_(y, y_hat, theta, .5))
print(reg_loss_(y, y_hat, theta, .05))
print(reg_loss_(y, y_hat, theta, .9))
