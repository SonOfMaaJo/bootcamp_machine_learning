import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mylinearregression import MyLinearRegression as MyLR


def plot(X, Y, Y_hat, **kwargs):
    cmap = mpl.colormaps[kwargs['color']]
    colors = cmap(np.linspace(0.4, 0.9, 2))
    plt.scatter(X, Y_hat, color=colors[0], marker='o',
                label=fr"{kwargs['label']}")
    plt.scatter(X, Y, color=colors[1], marker='o',
                label=fr"Predictive {kwargs['label']}", s=50)
    plt.xlabel(kwargs['xlabel'])
    plt.ylabel(kwargs['ylabel'])
    plt.legend(fontsize=7)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title(kwargs['title'])
    plt.show()


def predict_and_plot(X, Y, name, model, **kwargs):
    """
    function to predict and plot the results of the model.
    Args:
    _____
        X: vector of predictors of dimensions m * 1.
        Y: dependent variable. vector of dimensions m * 1.
        model: instance of class MylinearRegression
    Returns
        None
    """
    cmap = mpl.colormaps[kwargs['color']]
    colors = cmap(np.linspace(0.3, 0.9, 2))
    model.fit_(X, Y)
    y_hat = model.predict_(X)
    print(f"Error of model {name}: {model.mse_(Y, y_hat)}")
    plt.scatter(X, y_hat, color=colors[0], marker='o',
                label=fr"{kwargs['label']}")
    plt.scatter(X, Y, color=colors[1], marker='o',
                label=fr"Predictive {kwargs['label']}")
    plt.xlabel(kwargs['xlabel'])
    plt.ylabel(kwargs['ylabel'])
    plt.legend(fontsize=7)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title(kwargs['title'])
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("spacecraft_data.csv")
    Xage = np.array(data[['Age']])
    Xthrust = np.array(data[['Thrust_power']])
    Xdist = np.array(data[['Terameters']])
    X = np.array(data[['Age', 'Thrust_power', 'Terameters']])
    Y = np.array(data[['Sell_price']])

    myLR_age = MyLR(thetas=np.array([[1000.0], [-1.0]]),
                    alpha=2.5e-5, max_iter=100000)
    predict_and_plot(Xage, Y, 'MLR_age', myLR_age,
                     color='Greens', label='Sell price',
                     xlabel=r"$x_1$: age (in years)",
                     ylabel=r"$y$ sell price (in keuros)",
                     title=r"Sell price of spaceship in term of age")

    myLR_thrust = MyLR(thetas=np.array([[0.5], [2.0]]),
                       alpha=2.5e-5, max_iter=100000)
    predict_and_plot(Xthrust, Y, 'MLR_Thrust', myLR_thrust, color='Blues',
                     label='Sell price',
                     xlabel=r'$x_2$: thrust power (in 10km/s)',
                     ylabel=r'$y$: sell price (in Keuros)',
                     title="selling prices of spacecrafts if  f(thrust)")

    myLR_dist = MyLR(thetas=np.array([[800.0], [-1.0]]),
                     alpha=2.5e-5, max_iter=100000)
    predict_and_plot(Xdist, Y, 'MLR_dist', myLR_dist, color='Reds',
                     label='Sell price',
                     xlabel=r'$x_3$: distance totalizer value of spacecraft'
                     ' (in Tmeters)',
                     ylabel=r'$y$: sell price (in Keuros)',
                     title="selling prices of spacecrafts in f(distance)")

    my_lreg = MyLR(thetas=np.array([[1.0], [1.0], [1.0], [1.0]]),
                   alpha=9e-5, max_iter=500000)
    print("Error of before fitting : ",
          my_lreg.mse_(Y, my_lreg.predict_(X)))
    my_lreg.fit_(X, Y)
    print(my_lreg.thetas)
    print("Error after fitting : ",
          my_lreg.mse_(Y, my_lreg.predict_(X)))
    Y_hat = my_lreg.predict_(X)

    plot(Xage, Y, Y_hat, color='Greens', label='Sell price',
         xlabel=r"$x_1$: age (in years)",
         ylabel=r"$y$ sell price (in keuros)",
         title=r"Sell price of spaceship in term of age")

    plot(Xthrust, Y, Y_hat, color='Blues',
         label='Sell price',
         xlabel=r'$x_2$: thrust power (in 10km/s)',
         ylabel=r'$y$: sell price (in Keuros)',
         title="selling prices of spacecrafts if  f(thrust)")

    plot(Xdist, Y, Y_hat, color='Reds',
         label='Sell price',
         xlabel=r'$x_3$: distance totalizer value of spacecraft (in Tmeters)',
         ylabel=r'$y$: sell price (in Keuros)',
         title="selling prices of spacecrafts in f(distance)")
