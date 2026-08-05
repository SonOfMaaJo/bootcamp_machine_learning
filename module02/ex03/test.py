from gradient import gradient
import numpy as np


x = np.array([
    [-6, -7, -9],
    [13, -2, 14],
    [-7, 14, -1],
    [-8, -4, 6],
    [-5, -9, 6],
    [1, -5, 11],
    [9, -11, 8]])
y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
theta1 = np.array([0, 3, 0.5, -6]).reshape((-1, 1))
print(gradient(x, y, theta1))
theta2 = np.array([0, 0, 0, 0]).reshape((-1, 1))
print(gradient(x, y, theta2))
