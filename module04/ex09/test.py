import numpy as np
from my_logistic_regression import MyLogisticRegression as mylogr


theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
model1 = mylogr(theta, lambda_=5.0)
print(model1.penality, model1.lambda_)
model2 = mylogr(theta, penality=None)
print(model2.penality, model2.lambda_)
model3 = mylogr(theta, penality=None, lambda_=2.0)
print(model3.penality, model3.lambda_)
