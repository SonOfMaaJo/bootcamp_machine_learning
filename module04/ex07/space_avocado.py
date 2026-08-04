import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
from polynomial_model_extended import add_polynomial_features
from ridge import MyRidge as MLR
from data_spliter import data_spliter


def plot(x, y, tuple_, best_model, features):
    _, axes = plt.subplots(5, 3, figsize=(15, 15))
    cmap = mpl.colormaps["plasma"]
    colors = cmap(np.linspace(0, 1, 3))
    lambdas = np.arange(0, 1, 0.2)
    mean, std, power = tuple_
    x_ = (x - mean) / std
    for j in range(len(lambdas)):
        for i in range(3):
            best_model.set_params_(lambda_=lambdas[j])
            y_hat = best_model.predict_(
                add_polynomial_features(x_, power))
            axes[j][i].scatter(x[:, i].reshape(-1, 1), y, color=colors[i],
                               label='True Prices')
            axes[j][i].scatter(x[:, i].reshape(-1, 1), y_hat, color='red',
                               marker='.', label='predicitive prices')
            axes[j][i].set_xlabel(features[i])
            axes[j][i].set_ylabel('Price')
            axes[j][i].legend()

    plt.tight_layout()
    plt.show()


def get_models(models_list):
    models = []
    errors = []
    for model in models_list:
        params = list(model.values())[0]
        models.append(MLR(thetas=np.array(params['thetas']),
                          alpha=params['alpha'], lambda_=params['lambda'],
                          max_iter=100000))
        errors.append(params['loss'])
    return (models, errors)


if __name__ == "__main__":
    data = pd.read_csv("space_avocado.csv")
    features = ['weight', 'prod_distance', 'time_delivery']
    x = np.array(data[features])
    y = np.array(data[['target']])
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.7)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    mean = x_train.mean()
    std = x_train.std()

    with open("models.yml", "r", encoding="utf-8") as f:
        models_list = yaml.safe_load(f) or {}

    models, errors = get_models(models_list['models'])
    print(errors)
    plt.figure(figsize=(20, 10))
    plt.bar([f'model{i + 1}' for i in range(len(errors))], errors,
            color='skyblue', width=0.6)
    plt.title('Loss evaluations of the models.')
    plt.xlabel('models')
    plt.show()

    # models definition
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

    plot(x, y, (mean, std, power), best_model, features)
