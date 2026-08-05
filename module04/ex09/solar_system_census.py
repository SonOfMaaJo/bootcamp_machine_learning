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
    models = []
    errors = []
    class_ = []
    for model in models_list:
        params = list(model.values())[0]
        models.append(mylg(theta=np.array(params['thetas']),
                           alpha=params['alpha'], lambda_=params['lambda'],
                           max_iter=1000000))
        class_.append(params['class'])
        errors.append(params['loss'])
    return (models, errors, class_)


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
    mean = x_train.mean()
    std = x_train.std()

    with open("models.yml", "r", encoding="utf-8") as f:
        models_list = yaml.safe_load(f) or {}

    models, errors, class_ = get_models(models_list['models'])
    errors = np.array(errors).reshape(-1, 1)
    class_ = np.array(class_).reshape(-1, 1)
    models = np.array(models).reshape(-1, 1)
    model_class0 = models[class_ == 0]
    model_class1 = models[class_ == 1]
    model_class2 = models[class_ == 2]
    model_class3 = models[class_ == 3]
    errors_class0 = errors[class_ == 0]
    errors_class1 = errors[class_ == 1]
    errors_class2 = errors[class_ == 2]
    errors_class3 = errors[class_ == 3]
    models = []
    models.append(model_class0[np.unravel_index(np.argmin(errors_class0,
                                                          axis=None),
                                                errors_class0.shape)])
    models.append(model_class1[np.unravel_index(np.argmin(errors_class1,
                                                          axis=None),
                                                errors_class1.shape)])
    models.append(model_class2[np.unravel_index(np.argmin(errors_class2,
                                                          axis=None),
                                                errors_class2.shape)])
    models.append(model_class3[np.unravel_index(np.argmin(errors_class3,
                                                          axis=None),
                                                errors_class3.shape)])

    for i in range(4):
        print(f'training of the model{i}')
        models[i].fit_(x_test, (y_test == i).astype(int))
    class_ = classifier(x_test, models)
    index = errors.index(min(errors))
    best_model = models[index]
    print(f'the best model is the model{index}',
          f' with lambda={best_model.lambda_:.1f}')
    power = int((best_model.thetas.size - 1) / x.shape[1])
    x_ = add_polynomial_features((x_train - mean) / std, power)
    print('fitting of the best model...')
    best_model.fit_(x_, y_train)
    y_hat = best_model.predict_(
        add_polynomial_features((x_test - mean) / std, power))
    error = best_model.mse_(y_hat, y_test)
    print(f'Evaluation score of the best model on test set: {error}')

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
