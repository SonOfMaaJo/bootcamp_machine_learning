import pandas as pd
import numpy as np
import yaml
from polynomial_model_extended import add_polynomial_features
from my_logistic_regression import MyLogisticRegression as mylg
from data_spliter import data_spliter
from other_metrics import f1_score_

data = pd.read_csv("solar_system_census.csv")
origin = pd.read_csv("solar_system_census_planets.csv")
features = ['weight', 'height', 'bone_density']
x = np.array(data[features])
y = np.array(origin[['Origin']])
x_train, _, y_train, _ = data_spliter(x, y, 0.8)
y_train = y_train.reshape(-1, 1)
x_train, x_cross, y_train, y_cross = data_spliter(x_train, y_train, 0.7)
y_train = y_train.reshape(-1, 1)
y_cross = y_cross.reshape(-1, 1)
min_ = x_train.min(axis=0, keepdims=True)
max_ = x_train.max(axis=0, keepdims=True)

# models definition
lambdas = np.linspace(0, 1, 5)
number_row = 3 * x.shape[1] + 1
models: list = []
for lambda_ in lambdas:
    models.append(mylg(theta=np.zeros((number_row, 1)),
                       max_iter=1000000, lambda_=lambda_))
# fitting the models
mod: dict[str, list] = {}
with open("models.yml", "r", encoding="utf-8") as f:
    mod = yaml.safe_load(f) or {}
if "models" not in mod.keys():
    mod.update({"models": []})

x_ = add_polynomial_features((x_train - min_) / (max_ - min_), 3)
for i in range(4):
    for model in models:
        model.theta = np.zeros((number_row, 1))
        print(f'fitting of model{i + 1} with'
              f' lambda={model.lambda_:.1f} to discriminate class {i}...')
        model.fit_(x_, (y_train == i).astype(int))
        y_hat = model.predict_(
            add_polynomial_features((x_cross - min_) / (max_ - min_), 3)
        )
        mod["models"].append({
            f'model{i}{len(mod["models"])}': {
                "thetas": model.theta.tolist(),
                "alpha": float(model.alpha),
                "lambda": float(model.lambda_),
                "score": f1_score_((y_cross == i).astype(int),
                                  (y_hat >= 0.5).astype(int)),
                "class": i
            }
        })

with open("models.yml", "w", encoding="utf-8") as f:
    yaml.safe_dump(mod, f, sort_keys=False, allow_unicode=True)
