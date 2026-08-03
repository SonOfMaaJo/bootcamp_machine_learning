import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class Plotter:
    def __init__(self, ):
        pass

    def plot(x, y, y_hat, label, xlabel, ylabel, title):
        """Method to scatter plot points of data and the resulting predicting
        value.
        Args:
        -----
            x:  vectore of predictor
            y:  true value
            y_hat:  predicted value
            """
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
