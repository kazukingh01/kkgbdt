import optuna, json, argparse
import numpy as np
from functools import partial
from kkgbdt.tune import tune_parameter
from kklogger import set_logger
from kktestdata import DatasetRegistry


np.random.seed(0)
LOGGER = set_logger(__name__)
LIST_DATASEET = [
    "Fashion-MNIST","KDDCup99","SDSS17","covertype","gas-drift","helena","hiva_agnostic","mnist_784","nursery",
    "shuttle","splice","students_dropout_and_academic_success","tamilnadu-electricity","walking-activity"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter",  type=int, default=300)
    parser.add_argument("--trial", type=int, default=50)
    parser.add_argument("--jobs",  type=int, default=8)
    parser.add_argument("--dataset", type=lambda x: [int(y.strip()) for y in x.split(",")], default=",".join([str(i) for i in range(len(LIST_DATASEET))]))
    args = parser.parse_args()
    assert all(LIST_DATASEET[i] for i in args.dataset)

    reg      = DatasetRegistry()
    filepath = "params_cat.db"
    for dataset_name in [LIST_DATASEET[i] for i in args.dataset]:
        LOGGER.info(f"Tuning {dataset_name}...", color=["BOLD", "CYAN", "UNDERLINE"])
        dataset = reg.create(dataset_name)
        train_x, train_y, valid_x, valid_y = dataset.load_data(
            format="numpy", split_type="test", test_size=0.3
        )
        n_class = dataset.metadata.n_classes

        sampler = optuna.samplers.TPESampler()
        try: optuna.delete_study(study_name=f"{filepath.split('.')[0]}_{dataset_name}", storage=f'sqlite:///{filepath}')
        except KeyError: pass
        study   = optuna.create_study(study_name=f"{filepath.split('.')[0]}_{dataset_name}", storage=f'sqlite:///{filepath}', sampler=sampler, directions=["minimize"])
        func    = partial(tune_parameter,
            mode="cat", num_class=n_class, n_jobs=args.jobs, eval_string='model.evals_result["validation"]["MultiClass"][model.best_iteration]',
            x_train=train_x, y_train=train_y, loss_func="multi", num_iterations=args.iter,
            x_valid=valid_x, y_valid=valid_y, loss_func_eval="multi", sample_weight="balanced",
            early_stopping_rounds=5, early_stopping_idx=0,
            params_const = {
                "learning_rate"    : 0.05,
                "num_leaves"       : None,
                "is_gpu"           : False,
                "random_seed"      : 0,
                "max_depth"        : 8,
                "min_child_samples": None,
                "subsample"        : 1,
                "colsample_bytree" : None,
                "colsample_bynode" : None,  
                "max_bin"          : 128,
                "min_data_in_bin"  : None,
                "reg_alpha"        : None,
                "min_split_gain"   : None,
                "path_smooth"      : None,
                "verbosity"        : None,
                "min_child_weight" : None,
            },
            params_search='''{
                "colsample_bylevel" : trial.suggest_float("colsample_bylevel", 0.1, 0.9, log=False),
                "reg_lambda"        : trial.suggest_float("lambda", 1e-4, 1e3, log=True),
            }'''
        )
        study.optimize(func, n_trials=args.trial)
        dictwk  = {i: x.values[0] for i, x in enumerate(study.get_trials())}
        LOGGER.info(
            f"\nRESULT VALUES:\n{json.dumps({x:float(y) for x, y in dictwk.           items()}, indent=2)}\n" + 
            f"BEST PARAMETERS:\n{json.dumps({x:float(y) for x, y in study.best_params.items()}, indent=2)}",
            color=["BOLD", "GREEN"]
        )
