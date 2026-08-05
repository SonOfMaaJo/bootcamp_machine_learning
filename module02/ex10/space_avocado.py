import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR
from data_spliter import data_spliter


def plot(x, y, tuple_, best_model, features):
    _, axes = plt.subplots(1, 3, figsize=(15, 4))
    cmap = mpl.colormaps["plasma"]
    colors = cmap(np.linspace(0, 1, 3))
    mean, std, power, feature_index = tuple_
    x_ = (x[:, feature_index] - mean) / std
    y_hat = best_model.predict_(
        add_polynomial_features(x_.reshape(-1, 1), power)
    )
    for i in range(3):
        axes[i].scatter(x[:, i].reshape(-1, 1), y, color=colors[i], marker='.',
                        label='True Prices')
        axes[i].scatter(x[:, i].reshape(-1, 1), y_hat, color='red', marker='.',
                        label='predicitive prices')
        axes[i].set_xlabel(features[i])
        axes[i].set_ylabel('Price')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


def get_models(models_list):
    models = []
    errors = []
    features_index = []
    for model in models_list:
        params = list(model.values())[0]
        models.append(MyLR(thetas=np.array(params['thetas']),
                           alpha=params['alpha'],
                           max_iter=1000000))
        features_index.append(params['feature_index'])
        errors.append(params['loss'])
    return (models, errors, features_index)


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
    models, errors, features_index = get_models(models_list['models'])
    plt.figure(figsize=(20, 10))
    plt.bar([f'model{i + 1}' for i in range(len(errors))], errors,
            color='skyblue', width=0.6)
    plt.title('Loss evaluations of the models.')
    plt.xlabel('models')
    plt.show()

    # models definition
    index = errors.index(min(errors))
    best_model = models[index]
    feature_index = features_index[index]
    print(f'the best model is the model{index}',
          f' with feature={features[feature_index]}')
    power = int((best_model.thetas.size - 1) / x.shape[1])
    x_ = add_polynomial_features((x_train[:, feature_index] - mean)
                                 / std, power)
    print('fitting of the best model...')
    best_model.fit_(x_, y_train)
    y_hat = best_model.predict_(
        add_polynomial_features((x_test[:, feature_index] - mean) / std,
                                power)
    )
    error = best_model.mse_(y_hat, y_test)
    print(f'Evaluation score of the best model on test set: {error}')

    plot(x, y, (mean, std, power, feature_index),
         best_model, features)
