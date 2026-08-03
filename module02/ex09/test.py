from data_spliter import data_spliter
import numpy as np


import numpy as np
x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
print(data_spliter(x1, y, 0.8))
print(data_spliter(x1, y, 0.5))
x2 = np.array([[ 1, 42],
               [300, 10],
               [ 59, 1],
               [300, 59],
               [ 10, 42]])
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
print(data_spliter(x2, y, 0.8))
print(data_spliter(x2, y, 0.5))
