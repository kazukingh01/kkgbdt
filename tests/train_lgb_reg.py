import json
import numpy as np
import pandas as pd
from kktestdata import DatasetRegistry
from kkgbdt.model import KkGBDT, set_all_loglevel
from kkgbdt.loss import MSELoss, MAELoss
from kklogger import set_logger


np.random.seed(0)
LOGGER = set_logger(__name__)
set_all_loglevel("debug")


def rmse(y: np.ndarray, x: np.ndarray):
    return np.sqrt(((x - y) ** 2).mean())


if __name__ == "__main__":
    reg     = DatasetRegistry()
    dataset = reg.create("diamonds")
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataset.load_data(
        format="numpy", split_type="valid", test_size=0.3, valid_size=0.2
    )
    n_class = 1
    n_iter  = 100
    lr      = 0.2
    max_bin = 64
    ndepth  = -1
    valeval = {}
    LOGGER.info("data train", color=["BOLD", "CYAN"])
    LOGGER.info(f"\n{pd.Series(train_y).describe()}")
    LOGGER.info("data valid", color=["BOLD", "CYAN"])
    LOGGER.info(f"\n{pd.Series(valid_y).describe}")


    LOGGER.info("public loss rmse ( no validation )", color=["BOLD", "UNDERLINE", "GREEN"])
    model   = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func="reg", num_iterations=n_iter,
    )
    ndf_pred = model.predict(test_x, iteration_at=model.best_iteration)
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))

    LOGGER.info("public loss rmse", color=["BOLD", "UNDERLINE", "GREEN"])
    model   = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func="reg", num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["reg", MSELoss()], 
        early_stopping_rounds=20, early_stopping_idx=0,
    )
    ndf_pred = model.predict(test_x, iteration_at=model.best_iteration)
    assert model.best_iteration < n_iter
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    valeval["rmse_rmse"] = rmse(test_y, ndf_pred)

    LOGGER.info("public loss huber", color=["BOLD", "UNDERLINE", "GREEN"])
    model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func="huber", num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["huber", MSELoss()],
        early_stopping_rounds=20, early_stopping_idx=0, 
    )
    ndf_pred = model.predict(test_x)
    assert model.best_iteration < n_iter
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    valeval["huber_rmse"] = rmse(test_y, ndf_pred)

    LOGGER.info("custom loss MSELoss", color=["BOLD", "UNDERLINE", "GREEN"])
    model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func=MSELoss(), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["__copy__", MAELoss()],
        early_stopping_rounds=20, early_stopping_idx=0, 
    )
    ndf_pred = model.predict(test_x)
    assert model.best_iteration < n_iter
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    assert model.loss == KkGBDT.from_dict(model.to_dict()).loss
    valeval["MSELoss_rmse"] = rmse(test_y, ndf_pred)

    LOGGER.info(f"{json.dumps({x:float(y) for x, y in valeval.items()}, indent=2)}")
