import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR
from data_spliter import data_spliter


def plot(x, y, models, index, features):
    _, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        y_hat = models[index].predict_(
            add_polynomial_features(x[:, i], models[index].thetas.size - 1))
        axes[i].scatter(x[:, i], y, color='blue', label='True Prices')
        axes[i].scatter(x[:, i], y_hat, color='red', marker='.',
                        label='predicitive prices')
        axes[i].set_xlabel(features[i])
        axes[i].set_ylabel('Price')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("space_avocado.csv")
    features = ['weight', 'prod_distance', 'time_delivery']
    x = np.array(data[features])
    y = np.array(data[['target']])
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.7)

    # models definition
    theta4 = np.array([[-20], [160], [-80], [10], [-1]])
    models = [MyLR(thetas=np.array([[0], [0]]), alpha=0.5e-5,
                   max_iter=100000, name=' model1'),
              MyLR(thetas=np.array([[0], [0], [0], [0]]),
                   alpha=0.25e-20, max_iter=100000, name=' model2'),
              MyLR(thetas=np.array([[0], [0], [0], [0], [0]]),
                   alpha=0.25e-10,
                   max_iter=100000, name=' model3'),
              ]

    # fitting the models
    estimations = []
    errors = []
    for i in range(3):
        x_ = add_polynomial_features(x_train[:, i], i + 2 if i != 0 else 1)
        models[i].fit_(x_, y_train.reshape(-1, 1))
        estimations.append(models[i].predict_(
            add_polynomial_features(x_test[:, i], i + 2 if i != 0 else 1)))
        errors.append(models[i].mse_(estimations[i], y_test.reshape(-1, 1)))

    for i in range(3):
        print(f'Evaluation score of{models[i].name} : {errors[i]}')
    plt.bar([f'model{i + 1}' for i in range(3)], errors, color='skyblue',
            width=0.6)
    plt.title('MSE score evaluations metrics')
    plt.xlabel('models')
    plt.ylabel('MSE score')
    plt.show()

    plot(x, y, models, errors.index(min(errors)), features)
