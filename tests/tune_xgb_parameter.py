import os, optuna
import numpy as np
import shutil
from kkgbdt.tune import tune_parameter

if __name__ == "__main__":
    filepath = "test.db"
    if os.path.exists(filepath): os.remove(filepath)
    study = optuna.create_study(study_name="test", storage=f'sqlite:///test.db')
    num_class = 5
    x_train   = np.random.rand(1000, 100)
    y_train   = np.random.randint(0, num_class, 1000)
    x_valid   = np.random.rand(1000, 100)
    y_valid   = np.random.randint(0, num_class, 1000)
    from functools import partial
    func = partial(tune_parameter,
        mode="xgb", num_class=num_class, n_jobs=1, eval_string='model.evals_result["valid_0"]["mlogloss"][model.booster.best_iteration]',
        x_train=x_train, y_train=y_train, loss_func="multi:softmax", num_iterations=50,
        x_valid=x_valid, y_valid=y_valid, loss_func_eval="mlogloss", sample_weight="balanced",
        params_const = {
            "learning_rate"    : 0.03,
            "num_leaves"       : 100,
            "is_gpu"           : False,
            "random_seed"      : 0,
            "max_depth"        : 0,
            "min_child_samples": 20,
            "subsample"        : 1,
            "colsample_bylevel": 1,
            "colsample_bynode" : 1,
            "max_bin"          : 256,
            "min_data_in_bin"  : 5,
        },
        params_search='''{
            "min_child_weight" : trial.suggest_loguniform("min_child_weight", 1e-4, 1e3),
            "colsample_bytree" : trial.suggest_loguniform("colsample_bytree", 0.001, 0.5),
            "reg_alpha"        : trial.suggest_loguniform("alpha",  1e-4, 1e3),
            "reg_lambda"       : trial.suggest_loguniform("lambda", 1e-4, 1e3),
            "min_split_gain"   : trial.suggest_loguniform("gamma", 1e-10, 1.0),
        }'''
    )
    study.optimize(func, n_trials=100)
