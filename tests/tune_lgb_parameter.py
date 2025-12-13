import optuna, json, argparse
import numpy as np
from functools import partial
from kkgbdt.tune import tune_parameter
from kklogger import set_logger
from kktestdata import DatasetRegistry

# local imports
from exp_datasets import LIST_DATASEET, PARAMS_CONST_MODE


np.random.seed(0)
LOGGER = set_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter",  type=int, default=300)
    parser.add_argument("--trial", type=int, default=50)
    parser.add_argument("--jobs",  type=int, default=8)
    parser.add_argument("--dataset", type=lambda x: [int(y.strip()) for y in x.split(",")], default=",".join([str(i) for i in range(len(LIST_DATASEET))]))
    args = parser.parse_args()
    assert all(LIST_DATASEET[i] for i in args.dataset)

    reg      = DatasetRegistry()
    filepath = "params_lgb.db"
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
            mode="lgb", num_class=n_class, n_jobs=args.jobs, eval_string='model.booster.best_score["valid_0"]["multi_logloss"]',
            x_train=train_x, y_train=train_y, loss_func="multi", num_iterations=args.iter,
            x_valid=valid_x, y_valid=valid_y, loss_func_eval="multi", sample_weight="balanced",
            early_stopping_rounds=20, early_stopping_idx=0,
            params_const = PARAMS_CONST_MODE["lgb"],
            params_search='''{
                "min_child_weight" : trial.suggest_float("min_child_weight", 1e-4, 1e3, log=True),
                "colsample_bynode" : trial.suggest_float("colsample_bynode", 0.1, 0.9, log=False),
                "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 1e3, log=True),
            }'''
        )
        study.optimize(func, n_trials=args.trial)
        dictwk  = {i: x.values[0] for i, x in enumerate(study.get_trials())}
        LOGGER.info(
            f"\nRESULT VALUES:\n{json.dumps({x:float(y) for x, y in dictwk.           items()}, indent=2)}\n" + 
            f"BEST PARAMETERS:\n{json.dumps({x:float(y) for x, y in study.best_params.items()}, indent=2)}",
            color=["BOLD", "GREEN"]
        )
