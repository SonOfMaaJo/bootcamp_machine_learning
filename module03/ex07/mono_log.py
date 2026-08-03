import argparse
import numpy as np
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MLG
from tools import add_intercept
from data_spliter import data_spliter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='discriminate between citizens who come from a fav planet')
    parser.add_argument('-zipcode', type=int,
                        choices=[0, 1, 2, 3],
                        help='zip code of the fav planet')
    zipcode = parser.parse_args().zipcode
    data = pd.read_csv("solar_system_census.csv")
    to_planet = pd.read_csv("solar_system_census_planets.csv")
    x = np.array(data[['weight', 'height', 'bone_density']])
    from_ = np.array(to_planet[['Origin']])
    y = (from_ == zipcode).astype(int)
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.7)
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    mlr = MLG(np.array([[0], [0], [0], [0]]), alpha=0.25e-5, max_iter=100000)
    mlr.fit_(x_train, y_train)
    print(f'Value of theta after fitting : {mlr.theta}')
    y_hat = mlr.predict_(x_test)
    print(mlr.loss_elem_(y_test, y_hat))
