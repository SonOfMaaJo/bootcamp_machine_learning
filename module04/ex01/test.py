import numpy as np
from l2_reg import iterative_l2, l2

x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
print(iterative_l2(x))
print(l2(x))
y = np.array([3, 0.5, -6]).reshape((-1, 1))
print(iterative_l2(y))
print(l2(y))
