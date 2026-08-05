import pandas as pd
import numpy as np
import yaml
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR
from data_spliter import data_spliter


data = pd.read_csv("space_avocado.csv")
features = ['weight', 'prod_distance', 'time_delivery']
x = np.array(data[features])
y = np.array(data[['target']])
x_train, x_test, y_train, y_test = data_spliter(x, y, 0.7)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# models definition
models = [MyLR(thetas=np.array([[0], [0]]), alpha=0.5e-5,
               max_iter=1000000),
          MyLR(thetas=np.array([[0], [0], [0]]), alpha=0.5e-5,
               max_iter=1000000),
          MyLR(thetas=np.array([[0], [0], [0], [0]]),
               alpha=0.25e-5, max_iter=1000000),
          MyLR(thetas=np.array([[0], [0], [0], [0], [0]]),
               alpha=0.25e-5,
               max_iter=1000000),
          ]

# fitting the models
mod: dict[str, list] = dict()
with open("models.yml", "r", encoding="utf-8") as f:
    mod = yaml.safe_load(f) or {}
if "models" not in mod.keys():
    mod.update({"models": []})

for i in range(4):
    for j in range(3):
        mean = x_train[:, j].mean()
        std = x_train[:, j].std()
        x_ = add_polynomial_features((x_train[:, j] - mean) / std, i + 1)
        print(f'fitting of model{i + 1} with feature {features[j]}...')
        models[i].thetas = np.zeros(models[i].thetas.shape)
        models[i].fit_(x_, y_train)
        y_hat = models[i].predict_(
            add_polynomial_features((x_test[:, j] - mean) / std, i + 1)
        )
        mod["models"].append({
            f'model{i}{len(mod["models"])}': {
                "thetas": models[i].thetas.tolist(),
                "alpha": float(models[i].alpha),
                "loss": float(models[i].loss_(y_hat, y_test)),
                "feature_index": j
            }
        })

with open("models.yml", "w", encoding="utf-8") as f:
    yaml.safe_dump(mod, f, sort_keys=False, allow_unicode=True)
