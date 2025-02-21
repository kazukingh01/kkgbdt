import argparse, copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype 
from kkgbdt.model import KkGBDT
from kkgbdt.functions import log_loss
np.random.seed(0)


PARAMS_FIX = {
    "learning_rate": 0.2,
    "n_jobs": -1,
    "min_child_samples": 1,
    "subsample": 1.0,
    "colsample_bylevel": 1.0,
    "colsample_bynode": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1e-3,
    "min_split_gain": 0.0,
    "min_data_in_bin": 1,
    "min_child_weight": 1e-3,
    "colsample_bytree": 0.5,
    "path_smooth": 0.0,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--iter", type=int, default=300)
    args = parser.parse_args()
    PARAMS_FIX["n_jobs"] = args.jobs

    # data
    data    = fetch_covtype()
    train_x = data["data"  ][:-data["target"].shape[0]//5 ]
    train_y = data["target"][:-data["target"].shape[0]//5 ] - 1
    valid_x = data["data"  ][ -data["target"].shape[0]//5:]
    valid_y = data["target"][ -data["target"].shape[0]//5:] - 1
    n_class = np.unique(train_y).shape[0]
    df_eval = []

    # public loss
    loss_func = {
        "xgb": "multi:softmax",
        "lgb": "multiclass",
    }
    loss_func_eval = {
        "xgb": "mlogloss",
        "lgb": "multiclass",
    }
    is_softmax = {
        "xgb": True,
        "lgb": False,
    }
    list_params = []
    for max_bin in [32, 64, 128, 256]:
        for num_leaves in [50, 100, 200]:
            for max_depth in [-1, 6, 12]:
                for num_grad_quant_bins in [0, ]: # Temporarily, grad quant mode cannot be used [0, 4, 8, 16]
                    for random_seed in range(5):
                        for mode in ["xgb", "lgb"]:
                            list_params.append(
                                {"max_bin": max_bin, "num_leaves": num_leaves, "max_depth": max_depth, "mode": mode, "num_grad_quant_bins": num_grad_quant_bins, "random_seed": random_seed}
                            )
    for _param in tqdm(list_params):
        param = copy.deepcopy(PARAMS_FIX | _param)
        if param["num_grad_quant_bins"] > 0:
            param["use_quantized_grad"] = True
        else:
            param["use_quantized_grad"]  = False
            param["num_grad_quant_bins"] = None
        if param["use_quantized_grad"] and param["mode"] == "xgb":
            continue
        model = KkGBDT(n_class, **param)
        model.fit(
            train_x, train_y, loss_func=loss_func[param["mode"]], num_iterations=args.iter,
            x_valid=valid_x, y_valid=valid_y, loss_func_eval=loss_func_eval[param["mode"]], sample_weight="balanced"
        )
        se = pd.Series(param)
        se["time"] = model.time_train
        se["eval"] = log_loss(valid_y, model.predict(valid_x, is_softmax=is_softmax[param["mode"]], iteration_at=model.best_iteration))
        df_eval.append(se)
    df_eval = pd.concat(df_eval, axis=1, ignore_index=True).T
    df_eval.to_pickle("df_eval.pickle")
