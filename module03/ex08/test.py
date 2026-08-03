import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
    f1_score
from other_metrics import accuracy_score_, precision_score_, recall_score_, \
    f1_score_

y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
print(accuracy_score_(y, y_hat))
print(accuracy_score(y, y_hat))
print(precision_score_(y, y_hat))
print(precision_score(y, y_hat))
print(recall_score_(y, y_hat))
print(recall_score(y, y_hat))
print(f1_score_(y, y_hat))
print(f1_score(y, y_hat))
