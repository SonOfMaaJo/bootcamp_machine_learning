import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from my_logistic_regression import MyLogisticRegression as mlg
from data_spliter import data_spliter


def classifier(x, models):
    yhat = models[0].predict_(x)
    for i in range(1, 4):
        yhat = np.hstack((yhat, models[i].predict_(x)))
    return np.argmax(yhat, axis=1).reshape((-1, 1))


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
    data = pd.read_csv("solar_system_census.csv")
    origin = pd.read_csv("solar_system_census_planets.csv")
    x = np.array(data[['weight', 'height', 'bone_density']])
    y = np.array(origin[['Origin']])
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.6)
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    print(x_test.shape)
    models = [mlg(np.array([[-3.43], [-0.49], [-0.2], [-7.7]]),
                  alpha=0.25e-2, max_iter=1000000),
              mlg(np.array([[18.95], [-0.026], [-0.131], [8.35]]),
                  alpha=0.25e-2, max_iter=1000000),
              mlg(np.array([[-42.48], [0.05], [0.07], [11.57]]),
                  alpha=0.25e-2, max_iter=1000000),
              mlg(np.array([[2.33], [0.173], [-0.0190], [-20.04]]),
                  alpha=0.25e-2, max_iter=1000000)]
    for i in range(4):
        print(f'training of the model{i}')
        models[i].fit_(x_test, (y_test == i).astype(int))
        print('')

    class_ = classifier(x_test, models)
    print('percentage of correct prediction',
          f' : {class_[class_ == y_test].size / y_test.size * 100}%')

    class_ = classifier(x, models)
    plot(x[:, 0], y, class_, color='Greens', label='Origin',
         xlabel=r"$x_1$: weight",
         ylabel=r"$y$: Origin",
         title="Origin of the Citizens in term of weight")

    plot(x[:, 1], y, class_, color='Reds', label='Origin',
         xlabel=r"$x_1$: height",
         ylabel=r"$y$: Origin",
         title="Origin of the Citizens in term of height")

    plot(x[:, 2], y, class_, color='Blues', label='Origin',
         xlabel=r"$x_1$: bone_density",
         ylabel=r"$y$: Origin",
         title="Origin of the Citizens in term of bone_density")
