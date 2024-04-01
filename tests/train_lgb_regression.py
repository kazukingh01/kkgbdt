import numpy as np
from sklearn.datasets import fetch_california_housing
from kkgbdt.model import KkGBDT
from kkgbdt.loss import HuberLoss, MSELoss
np.random.seed(0)


def rsme(x: np.ndarray, t: np.ndarray):
    return ((x - t) ** 2).mean()

if __name__ == "__main__":
    data = fetch_california_housing()
    train_x = data["data"  ][:-data["target"].shape[0]//5 ]
    train_y = data["target"][:-data["target"].shape[0]//5 ]
    valid_x = data["data"  ][ -data["target"].shape[0]//5:]
    valid_y = data["target"][ -data["target"].shape[0]//5:]
    # public loss
    model   = KkGBDT(1, mode="lgb", max_bin=64)
    model.fit(
        train_x, train_y, loss_func="huber", num_iterations=100,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["huber"], 
    )
    print(model.predict(valid_x, is_softmax=False))
    print(f"best iteration the model.booster have is: {model.booster.best_iteration}")
    print(rsme(model.predict(valid_x, is_softmax=False), valid_y))
    # public loss with "use_quantized_grad"
    model = KkGBDT(1, mode="lgb", max_bin=64, use_quantized_grad=True, num_grad_quant_bins=8, )
    model.fit(
        train_x, train_y, loss_func="huber", num_iterations=100,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["huber"], 
    )
    print(model.predict(valid_x, is_softmax=False))
    print(f"best iteration the model.booster have is: {model.booster.best_iteration}")
    print(rsme(model.predict(valid_x, is_softmax=False), valid_y))
    # custom loss HuberLoss
    model = KkGBDT(1, mode="lgb", max_bin=64)
    model.fit(
        train_x, train_y, loss_func=HuberLoss(beta=10), num_iterations=100,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=[HuberLoss(beta=10), "rmse"],
        early_stopping_rounds=20, early_stopping_name=0, 
    )
    print(model.predict(valid_x, is_softmax=False))
    print(f"best iteration the model.booster have is: {model.booster.best_iteration}")
    print(MSELoss()(model.predict(valid_x, is_softmax=False), valid_y))
    print(rsme(model.predict(valid_x, is_softmax=False), valid_y))
