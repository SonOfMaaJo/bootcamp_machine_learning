from loss import loss_
import numpy as np


X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
print(loss_(X, Y))
print(loss_(X, X))
