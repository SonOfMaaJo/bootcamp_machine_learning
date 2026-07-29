import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from other_losses import mse_, rmse_, mae_, r2score_


x = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
print(mse_(x,y))
print(mean_squared_error(x,y))
print(rmse_(x,y))
print(sqrt(mean_squared_error(x,y)))
print(mae_(x,y))
print(mean_absolute_error(x,y))
print(r2score_(x,y))
print(r2_score(x,y))
