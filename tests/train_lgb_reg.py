import json
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from kkgbdt.model import KkGBDT, set_all_loglevel
from kkgbdt.loss import MSELoss, MAELoss
from kklogger import set_logger


np.random.seed(0)
LOGGER = set_logger(__name__)
set_all_loglevel("debug")


def rmse(y: np.ndarray, x: np.ndarray):
    return ((x - y) ** 2).mean()


if __name__ == "__main__":
    data    = fetch_california_housing()
    train_x = data["data"  ][:-data["target"].shape[0]//5 ]
    train_y = data["target"][:-data["target"].shape[0]//5 ]
    valid_x = data["data"  ][ -data["target"].shape[0]//5:]
    valid_y = data["target"][ -data["target"].shape[0]//5:]
    n_class = 1
    n_iter  = 500
    lr      = 0.05
    max_bin = 64
    ndepth  = -1
    valeval = {}
    LOGGER.info("data train", color=["BOLD", "CYAN"])
    LOGGER.info(f"\n{pd.Series(train_y).describe()}")
    LOGGER.info("data valid", color=["BOLD", "CYAN"])
    LOGGER.info(f"\n{pd.Series(valid_y).describe}")

    LOGGER.info("Test public loss", color=["BOLD", "UNDERLINE", "GREEN"])
    model   = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    LOGGER.info("train without validation", color=["BOLD", "CYAN"])
    model.fit(train_x, train_y, loss_func="huber", num_iterations=10, sample_weight=np.random.rand(train_x.shape[0]))
    LOGGER.info("train with validation", color=["BOLD", "CYAN"])
    model.fit(
        train_x, train_y, loss_func="rmse", num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["rmse", MSELoss()], 
        early_stopping_rounds=20, early_stopping_name=0,
    )
    ndf_pred = model.predict(valid_x, iteration_at=model.best_iteration)
    valeval["rmse_rmse"] = rmse(valid_y, ndf_pred)

    LOGGER.info("custom loss huber", color=["BOLD", "UNDERLINE", "GREEN"])
    model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func="huber", num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["huber", MSELoss()],
        early_stopping_rounds=20, early_stopping_name=0, 
    )
    ndf_pred = model.predict(valid_x)
    valeval["huber_rmse"] = rmse(valid_y, ndf_pred)

    LOGGER.info("custom loss MSELoss", color=["BOLD", "UNDERLINE", "GREEN"])
    model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func=MSELoss(), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["__copy__", MAELoss()],
        early_stopping_rounds=20, early_stopping_name=0, 
    )
    ndf_pred = model.predict(valid_x)
    valeval["MSELoss_rmse"] = rmse(valid_y, ndf_pred)

    LOGGER.info(f"{json.dumps({x:float(y) for x, y in valeval.items()}, indent=2)}")
