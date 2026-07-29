import numpy as np
from prediction import predict_
from loss import loss_elem_, loss_

x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
print(loss_elem_(y1, y_hat1))
print(loss_(y1, y_hat1))
