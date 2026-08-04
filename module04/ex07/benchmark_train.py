import pandas as pd
import numpy as np
import yaml
from polynomial_model_extended import add_polynomial_features
from my_logistic_regression import MyLogisticRegression as mlrg
from data_spliter import data_spliter
from other_metrics import f1_score_


def classifier(x, models):
    yhat = models[0].predict_(x)
    for i in range(1, len(models)):
        yhat = np.hstack((yhat, models[i].predict_(x)))
    return np.argmax(yhat, axis=1).reshape((-1, 1))


data = pd.read_csv("space_avocado.csv")
features = ['weight', 'prod_distance', 'time_delivery']
x = np.array(data[features])
y = np.array(data[['target']])
x_train, _, y_train, _ = data_spliter(x, y, 0.8)
y_train = y_train.reshape(-1, 1)
x_train, x_cross, y_train, y_cross = data_spliter(x_train, y_train, 0.7)
y_train = y_train.reshape(-1, 1)
y_cross = y_cross.reshape(-1, 1)
mean = x_train.mean()
std = x_train.std()

lambdas = np.linspace(0, 1, 20).tolist()
# models definition
models: list = []
for lambda_ in lambdas:
    models.append(mlrg(thetas=np.array([[0] for _ in
                                        range(x.shape[1] * 3 + 1)]),
                       alpha=0.25e-2,
                       max_iter=1000000, penality='l2',
                       lambda_=lambda_))

# fitting the models
mod: dict[str, list] = dict()
mod.update({"models": []})

for i in range(len(models)):
    x_ = add_polynomial_features((x_train - mean) / std, 3)
    print(f'fitting of model{i + 1} with'
          f' lambda={models[i].lambda_:.1f}...')
    models[i].fit_(x_, y_train)
    y_hat = models[i].predict_(
        add_polynomial_features((x_cross - mean) / std, 3)
    )
    mod["models"].append({
        f'model{i}{len(mod["models"])}': {
            "thetas": models[i].thetas.tolist(),
            "alpha": float(models[i].alpha),
            "lambda": float(models[i].lambda_),
            "loss": float(models[i].loss_(y_hat, y_cross))
        }
    })

with open("models.yml", "w", encoding="utf-8") as f:
    yaml.safe_dump(mod, f, sort_keys=False, allow_unicode=True)
