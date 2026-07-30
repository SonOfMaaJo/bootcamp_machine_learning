import numpy as np
from z_score import zscore


X = np.array([0, 15, -9, 7, 12, 3, -21])
print(zscore(X))
Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
print(zscore(Y))
