import pandas as pd
import numpy as np
import yaml
from polynomial_model_extended import add_polynomial_features
from ridge import MyRidge as MLR
from data_spliter import data_spliter
from z_score import zscore


data = pd.read_csv("solar_system_census.csv")
origin = pd.read_csv("solar_system_census_planets.csv")
features = ['weight', 'prod_distance', 'bone_density']
x = np.array(data[features])
y = np.array(origin[['Origin']])
x_train, _, y_train, _ = data_spliter(x, y, 0.8)
y_train = y_train.reshape(-1, 1)
x_train, x_cross, y_train, y_cross = data_spliter(x_train, y_train, 0.7)
y_train = y_train.reshape(-1, 1)
y_cross = y_cross.reshape(-1, 1)
mean = x_train.mean()
std = x_train.std()

# models definition
models = [MLR(thetas=np.array([[-1] for _ in range(x.shape[1])]),
              alpha=0.25e-2, max_iter=1000000),
          MLR(thetas=np.array([[0] for _ in range(x.shape[1] * 2)]),
              alpha=0.25e-2, max_iter=1000000),
          MLR(thetas=np.array([[0] for _ in range(x.shape[1] * 3)]),
              alpha=0.25e-2, max_iter=1000000),
          MLR(thetas=np.array([[0] for _ in range(x.shape[1] * 4)]),
              alpha=0.25e-2, max_iter=1000000)
          ]
lambdas = np.arange(0, 1, 0.2)

# fitting the models
mod: dict[str, list] = dict()
mod.update({"models": []})

for i in range(4):
    x_ = add_polynomial_features((x_train - mean) / std, i + 1)
    for lambda_ in lambdas:
        models[i].set_params_(
            thetas=np.array(
                [[0] for _ in range(x.shape[1] * (i + 1) + 1)]),
            lambda_=lambda_)
        print(f'fitting of model{i + 1} with'
              f' lambda={models[i].lambda_:.1f}...')
        models[i].fit_(x_, y_train)
        y_hat = models[i].predict_(
            add_polynomial_features((x_cross - mean) / std, i + 1)
        )
        thetas, alpha, _, lambda_ = models[i].get_params_()
        mod["models"].append({
            f'model{i}{len(mod["models"])}': {
                "thetas": thetas.tolist(),
                "alpha": float(alpha),
                "lambda": float(lambda_),
                "loss": float(models[i].loss_(y_hat, y_cross))
            }
        })

with open("models.yml", "w", encoding="utf-8") as f:
    yaml.safe_dump(mod, f, sort_keys=False, allow_unicode=True)
