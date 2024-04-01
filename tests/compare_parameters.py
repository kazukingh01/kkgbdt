import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype 
from kkgbdt.model import KkGBDT


def log_loss(y: np.ndarray, x: np.ndarray):
    assert isinstance(y, np.ndarray) and len(y.shape) == 1
    assert isinstance(x, np.ndarray) and len(x.shape) == 2
    assert y.dtype in [int, np.int8, np.int16, np.int32, np.int64]
    ndf = x[np.arange(x.shape[0]), y]
    return (-1 * np.log(ndf)).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    data    = fetch_covtype()
    train_x = data["data"  ][:-data["target"].shape[0]//5 ]
    train_y = data["target"][:-data["target"].shape[0]//5 ] - 1
    valid_x = data["data"  ][ -data["target"].shape[0]//5:]
    valid_y = data["target"][ -data["target"].shape[0]//5:] - 1
    n_class = np.unique(train_y).shape[0]
    df_eval = []

    # public loss
    params_fix = {
        "learning_rate": 0.2,
        "n_jobs": args.jobs,
        "min_child_samples": 1,
        "subsample": 1.0,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1e-3,
        "min_split_gain": 0.0,
        "min_data_in_bin": 1,
        "min_child_weight": 1e-3,
    }
    for max_bin in [32, 64, 128, 256]:
        for num_leaves in [50, 100, 200]:
            for max_depth in [-1, 5, 10]:
                for colsample_bytree in [0.1, 0.3, 0.6, 1.0]:
                    for random_seed in range(20):
                        param = (params_fix | {"max_bin": max_bin, "num_leaves": num_leaves, "max_depth": max_depth, "colsample_bytree": colsample_bytree, "random_seed": random_seed})
                        model = KkGBDT(n_class, mode="xgb", **param)
                        model.fit(
                            train_x, train_y, loss_func="multi:softmax", num_iterations=300,
                            x_valid=valid_x, y_valid=valid_y, loss_func_eval=["mlogloss"], sample_weight="balanced"
                        )
                        se = pd.Series(param)
                        se["time_xgb"] = model.time_train
                        se["eval_xgb"] = log_loss(valid_y, model.predict(valid_x, is_softmax=True))
                        model = KkGBDT(n_class, mode="lgb", **param)
                        model.fit(
                            train_x, train_y, loss_func="multiclass", num_iterations=300,
                            x_valid=valid_x, y_valid=valid_y, loss_func_eval=["multiclass"], sample_weight="balanced"
                        )
                        se["time_lgb"] = model.time_train
                        se["eval_lgb"] = log_loss(valid_y, model.predict(valid_x, is_softmax=False))
                        df_eval.append(se)
    df_eval = pd.concat(df_eval, axis=1, ignore_index=True).T
    df_eval.to_pickle("df_eval.pickle")
