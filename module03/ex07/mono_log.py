import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from my_logistic_regression import MyLogisticRegression as mlg
from data_spliter import data_spliter
import sys


def plot(X, Y, Y_hat, **kwargs):
    cmap = mpl.colormaps[kwargs['color']]
    colors = cmap(np.linspace(0.4, 0.9, 2))
    plt.scatter(X, Y, color=colors[0], marker='o',
                label=fr"{kwargs['label']}", s=52)
    plt.scatter(X, Y_hat, color=colors[1], marker='o',
                label=fr"Predictive {kwargs['label']}", s=50, alpha=0.5)
    plt.xlabel(kwargs['xlabel'])
    plt.ylabel(kwargs['ylabel'])
    plt.legend(fontsize=7)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title(kwargs['title'])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='discriminate between citizens who come from a fav planet')
    parser.add_argument('-zipcode', type=int, required=True,
                        choices=[0, 1, 2, 3],
                        help='zip code of the fav planet')
    zipcode = parser.parse_args().zipcode
    data = pd.read_csv("solar_system_census.csv")
    to_planet = pd.read_csv("solar_system_census_planets.csv")
    x = np.array(data[['weight', 'height', 'bone_density']])
    from_ = np.array(to_planet[['Origin']])
    y = (from_ == zipcode).astype(int)
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.6)
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    mlr = mlg(np.array([[0], [0], [0], [0]]), alpha=0.25e-2, max_iter=1000000)
    print('fitting...')
    mlr.fit_(x_train, y_train)
    print(f'Value of theta after fitting : {mlr.theta}')
    y_hat = mlr.predict_(x_test)
    p_label = (y_hat >= 0.5).astype(int)
    print("percentage of total correct prediction"
          f" : {p_label[p_label == y_test].size / y_test.size * 100}%")

    y_hat = mlr.predict_(x)
    plot(x[:, 0], y, y_hat, color='Greens', label='Origin',
         xlabel=r"$x_1$: weight",
         ylabel=r"$y$: Origin",
         title="Origin of the Citizens in term of weight")

    plot(x[:, 1], y, y_hat, color='Reds', label='Origin',
         xlabel=r"$x_1$: height",
         ylabel=r"$y$: Origin",
         title="Origin of the Citizens in term of height")

    plot(x[:, 2], y, y_hat, color='Blues', label='Origin',
         xlabel=r"$x_1$: bone_density",
         ylabel=r"$y$: Origin",
         title="Origin of the Citizens in term of bone_density")
