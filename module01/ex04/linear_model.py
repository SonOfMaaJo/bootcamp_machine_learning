import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR
from tools import add_intercept


def loss_function(theta0, y, x):
    """calcul of the loss function in function of theta"""
    return lambda theta: np.array([(1 / (2 * y.size)) *
                                   np.sum((add_intercept(x) @ np.array(
                                       [[theta0], [theta[i]]]) - y) ** 2)
                                   for i in range(theta.size)])


data = pd.read_csv("are_blue_pills_magics.csv")
Xpill = np.array(data['Micrograms']).reshape(-1, 1)
Yscore = np.array(data['Score']).reshape(-1, 1)
nfunc = 7
thetas_ = np.linspace(-8, 8, nfunc)
theta = np.linspace(1, 16, 500)
linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))

linear_model1.fit_(Xpill, Yscore)
Ymodel1 = linear_model1.predict_(Xpill)

linear_model2.fit_(Xpill, Yscore)
Ymodel2 = linear_model2.predict_(Xpill)

print(MyLR.mse_(Yscore, Ymodel1))
print(mean_squared_error(Yscore, Ymodel1))
print(MyLR.mse_(Yscore, Ymodel2))
print(mean_squared_error(Yscore, Ymodel2))


plt.plot(Xpill, Ymodel1, 'g-x', label=r"$S_{predict}(pills)$", linewidth=2)
plt.plot(Xpill, Yscore, 'bo', label=r"$S_{true}(pills)$", linewidth=2)
plt.xlabel("Quantity of the blue pill (in micrograms)")
plt.ylabel("Space driving score")
plt.legend(loc="upper right", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.title("piloting score versus the quantity of blue pills")
plt.show()

cmap = mpl.colormaps['Greys']
colors_func = cmap(np.linspace(0.3, 0.9, nfunc))
for i in range(nfunc - 1):
    y = loss_function(thetas_[i], Yscore, Xpill)(theta)
    plt.plot(theta, y,
             label=fr"$J(\theta_0=c_{i}, \theta_1)$", color=colors_func[i],
             linewidth=2)
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"cost function $J(\theta_0, \theta_1)$")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.6)
plt.title("The loss function J(theta) in function of the value theta")

plt.tight_layout()
plt.show()
