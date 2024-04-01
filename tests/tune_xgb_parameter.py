import os, optuna
import numpy as np
from sklearn.datasets import fetch_covtype 
from kkgbdt.tune import tune_parameter

if __name__ == "__main__":
    filepath = "test.db"
    if os.path.exists(filepath): os.remove(filepath)
    study   = optuna.create_study(study_name="test", storage=f'sqlite:///test.db')
    data    = fetch_covtype()
    train_x = data["data"  ][:-data["target"].shape[0]//5 ]
    train_y = data["target"][:-data["target"].shape[0]//5 ] - 1
    valid_x = data["data"  ][ -data["target"].shape[0]//5:]
    valid_y = data["target"][ -data["target"].shape[0]//5:] - 1
    n_class = np.unique(train_y).shape[0]
    from functools import partial
    func = partial(tune_parameter,
        mode="xgb", num_class=n_class, n_jobs=1, eval_string='model.evals_result["valid_0"]["mlogloss"][model.booster.best_iteration]',
        x_train=train_x, y_train=train_y, loss_func="multi:softmax", num_iterations=50,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval="mlogloss", sample_weight="balanced",
        early_stopping_rounds=5, early_stopping_name=0,
        params_const = {
            "learning_rate"    : 0.1,
            "num_leaves"       : 100,
            "is_gpu"           : False,
            "random_seed"      : 0,
            "max_depth"        : 0,
            "min_child_samples": 20,
            "subsample"        : 1,
            "colsample_bylevel": 1,
            "colsample_bynode" : 1,
            "max_bin"          : 64,
            "min_data_in_bin"  : 5,
        },
        params_search='''{
            "min_child_weight" : trial.suggest_float("min_child_weight", 1e-4, 1e3, log=True),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.001, 0.5, log=True),
            "reg_alpha"        : trial.suggest_float("alpha",  1e-4, 1e3, log=True),
            "reg_lambda"       : trial.suggest_float("lambda", 1e-4, 1e3, log=True),
            "min_split_gain"   : trial.suggest_float("gamma", 1e-10, 1.0, log=True),
        }'''
    )
    study.optimize(func, n_trials=10)
