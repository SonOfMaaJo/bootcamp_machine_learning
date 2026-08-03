import numpy as np
from confusion_matrix import confusion_matrix_


y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])
print(confusion_matrix_(y, y_hat))
print(confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))
