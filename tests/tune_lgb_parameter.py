import os, optuna, json, argparse
import numpy as np
from functools import partial
from kkgbdt.tune import tune_parameter
from kklogger import set_logger
from kktestdata import DatasetRegistry


np.random.seed(0)
LOGGER = set_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter",  type=int, default=300)
    parser.add_argument("--ntrial", type=int, default=50)
    args = parser.parse_args()

    reg      = DatasetRegistry()
    filepath = "params.db"
    for dataset_name in [
        "Fashion-MNIST","KDDCup99","SDSS17","covertype","gas-drift","helena","hiva_agnostic","mnist_784","nursery",
        "shuttle","splice","students_dropout_and_academic_success","tamilnadu-electricity","walking-activity"
    ]:
        dataset = reg.create(dataset_name)
        train_x, train_y, valid_x, valid_y = dataset.load_data(
            format="numpy", split_type="test", test_size=0.3
        )
        n_class = dataset.metadata.n_classes

        sampler = optuna.samplers.TPESampler()
        try: optuna.delete_study(study_name=f"params_lgb_{dataset_name}", storage=f'sqlite:///params.db')
        except KeyError: pass
        study   = optuna.create_study(study_name=f"params_lgb_{dataset_name}", storage=f'sqlite:///params.db', sampler=sampler, directions=["minimize"])
        func    = partial(tune_parameter,
            mode="lgb", num_class=n_class, n_jobs=-1, eval_string='model.booster.best_score["valid_0"]["multi_logloss"]',
            x_train=train_x, y_train=train_y, loss_func="multi", num_iterations=args.niter,
            x_valid=valid_x, y_valid=valid_y, loss_func_eval="multi", sample_weight="balanced",
            early_stopping_rounds=5, early_stopping_idx=0,
            params_const = {
                "learning_rate"    : 0.05,
                "num_leaves"       : (2 ** 8), # depth = 8
                "is_gpu"           : False,
                "random_seed"      : 0,
                "max_depth"        : -1,
                "min_child_samples": None,
                "subsample"        : 1,
                "colsample_bytree" : 1,
                "colsample_bylevel": 1,
                "max_bin"          : 128,
                "min_data_in_bin"  : None,
                "reg_alpha"        : None,
                "min_split_gain"   : None,
                "path_smooth"      : None,
                "verbosity"        : None,
            },
            params_search='''{
                "min_child_weight" : trial.suggest_float("min_child_weight", 1e-4, 1e3, log=True),
                "colsample_bynode" : trial.suggest_float("colsample_bynode", 0.1, 0.9, log=False),
                "reg_lambda"       : trial.suggest_float("lambda", 1e-4, 1e3, log=True),
            }'''
        )
        study.optimize(func, n_trials=args.ntrial)
        dictwk  = {i: x.values[0] for i, x in enumerate(study.get_trials())}
        LOGGER.info(
            f"\nRESULT VALUES:\n{json.dumps({x:float(y) for x, y in dictwk.           items()}, indent=2)}\n" + 
            f"BEST PARAMETERS:\n{json.dumps({x:float(y) for x, y in study.best_params.items()}, indent=2)}",
            color=["BOLD", "GREEN"]
        )
