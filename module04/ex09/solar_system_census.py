import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
from polynomial_model_extended import add_polynomial_features
from my_logistic_regression import MyLogisticRegression as mylg
from data_spliter import data_spliter
from other_metrics import f1_score_


def get_models(models_list):
    models_ = []
    scores_ = []
    c_ = []
    for model in models_list:
        params = list(model.values())[0]
        models_.append(mylg(theta=np.array(params['thetas']),
                            alpha=params['alpha'], lambda_=params['lambda'],
                            max_iter=1000000))
        c_.append(params['class'])
        scores_.append(params['score'])
    return (np.array(models_).reshape(-1, 1), np.array(scores_).reshape(-1, 1),
            np.array(c_).reshape(-1, 1))


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


def classifier(x, models):
    yhat = models[0].predict_(x)
    for i in range(1, 4):
        yhat = np.hstack((yhat, models[i].predict_(x)))
    return np.argmax(yhat, axis=1).reshape((-1, 1))


if __name__ == "__main__":
    data = pd.read_csv("solar_system_census.csv")
    origin = pd.read_csv("solar_system_census_planets.csv")
    features = ['weight', 'height', 'bone_density']
    x = np.array(data[features])
    y = np.array(origin[['Origin']])
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.7)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    min_ = x_train.min(axis=0, keepdims=True)
    max_ = x_train.max(axis=0, keepdims=True)

    with open("models.yml", "r", encoding="utf-8") as f:
        models_list = yaml.safe_load(f) or {}

    models, scores, class_ = get_models(models_list['models'])
    models_: list = []
    scores_: list = []
    for i in range(4):
        models_.append(models[class_ == i])
        scores_.append(scores[class_ == i])
    models = []
    x_ = add_polynomial_features((x_train - min_) / (max_ - min_), 3)
    for i in range(4):
        models.append(models_[i][np.unravel_index(np.argmax(scores_[i],
                                                            axis=None),
                                                  scores_[i].shape)])
        print(f'training of the model{i}')
        models[i].fit_(x_, (y_train == i).astype(int))
    x_ = add_polynomial_features((x_test - min_) / (max_ - min_), 3)
    class_ = classifier(x_, models)
    for i in range(4):
        print('Evaluation score (f1score) of the best model on test set to'
              f' discriminate class {i}: {f1_score_(y_test, class_, i)}')
    x_ = add_polynomial_features((x - min_) / (max_ - min_), 3)
    class_ = classifier(x_, models)
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
