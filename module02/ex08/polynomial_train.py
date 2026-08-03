import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR
from z_score import zscore


def plot(x, y, models):
    cmap = mpl.colormaps['Greens']
    colors = cmap(np.linspace(0.4, 0.9, 6))
    continuous_x = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)
    for i in range(6):
        x_p = add_polynomial_features(continuous_x, i + 1)
        y_hat = models[i].predict_(x_p)
        plt.plot(continuous_x, y_hat, color=colors[i],
                 label=f"{models[i].name}")
        plt.scatter(x, y, color=colors[i], marker='o',
                    label="True Value")
    plt.xlabel("Quantity of blue pill (in micrograms)")
    plt.ylabel("Space driving score")
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.6)
    plt.title("Space driving score as a function of the quantity of blue pill")
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("are_blue_pills_magics.csv")
    x = np.array(data[['Micrograms']])
    y = np.array(data[['Score']])
    theta4 = np.array([[-20], [160], [-80], [10], [-1]])
    theta5 = np.array([[1140], [-1850], [1110], [-305], [40],
                       [-2]])
    theta6 = np.array([[9110], [-18015], [13400], [-4935], [966], [-96.4],
                       [3.86]])
    models = [MyLR(thetas=np.array([[-20], [1]]), alpha=2.5e-5,
                   max_iter=100000, name=' model1'),
              MyLR(thetas=np.array([[50], [-29], [50]]),
                   alpha=0.0005e-5, max_iter= 100000, name=' model2'),
              MyLR(thetas=np.array([[100], [-445], [-66], [17]]),
                   alpha=0.0005e-5,
                   max_iter=100000, name=' model3'),
              MyLR(thetas=theta4,
                   alpha=0.0005e-6,
                   max_iter=100000, name=' model4'),
              MyLR(thetas=theta5,
                   alpha=0.0005e-6,
                   max_iter=100000, name=' model5'),
              MyLR(thetas=theta6,
                   alpha=0.0005e-6,
                   max_iter=100000, name=' model6')
              ]
    estimations = []
    errors = []
    for i in range(6):
        x_ = add_polynomial_features(x, i + 1)
        models[i].fit_(x_, y)
        estimations.append(models[i].predict_(x_))
        errors.append(models[i].mse_(estimations[i], y))
    for i in range(6):
        print(f'Evaluation score of{models[i].name} : {errors[i]}')
    plt.bar([i + 1 for i in range(6)], errors, color='skyblue', width=0.6)
    plt.title('MSE score in funciton of polynomial degree')
    plt.xlabel('Polynomial degree')
    plt.ylabel('MSE score')
    plt.show()

    plot(x, y, models)
