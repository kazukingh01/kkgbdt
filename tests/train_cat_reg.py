import numpy as np
from sklearn.datasets import fetch_california_housing
from kkgbdt.model import KkGBDT
from kkgbdt.loss import MSELoss
from kklogger import set_logger
import argparse


np.random.seed(0)
LOGGER = set_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="enable GPU for catboost")
    args = parser.parse_args()

    data = fetch_california_housing()
    train_x = data["data"  ][:-data["target"].shape[0]//5 ]
    train_y = data["target"][:-data["target"].shape[0]//5 ]
    valid_x = data["data"  ][ -data["target"].shape[0]//5:]
    valid_y = data["target"][ -data["target"].shape[0]//5:]
    n_iter  = 150
    lr      = 0.1
    max_bin = 128
    ndepth  = -1

    model = KkGBDT(1, mode="cat", learning_rate=lr, max_bin=max_bin, max_depth=ndepth, is_gpu=args.gpu)
    model.fit(
        train_x, train_y, loss_func="RMSE", num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval="RMSE",
        early_stopping_rounds=20, early_stopping_name=0,
    )
    pred = model.predict(valid_x, is_softmax=False)
    LOGGER.info(f"rmse(normal): {np.sqrt(((pred - valid_y) ** 2).mean())}", color=["GREEN", "BOLD"])

    loss_custom = MSELoss()
    model_custom = KkGBDT(1, mode="cat", learning_rate=lr, max_bin=max_bin, max_depth=ndepth, is_gpu=args.gpu)
    model_custom.fit(
        train_x, train_y, loss_func=loss_custom, num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=[loss_custom],
        early_stopping_rounds=20, early_stopping_name=0,
    )
    pred_custom = model_custom.predict(valid_x, is_softmax=False)
    LOGGER.info(f"rmse(custom): {np.sqrt(((pred_custom - valid_y) ** 2).mean())}", color=["GREEN", "BOLD"])
