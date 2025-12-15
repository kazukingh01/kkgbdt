import optuna, json, argparse
import numpy as np
import pandas as pd
from functools import partial
from kkgbdt.model import KkGBDT
from kkgbdt.tune import tune_parameter
from kklogger import set_logger
from kktestdata import DatasetRegistry
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


LOGGER = set_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",             type=str,   required=True)
    parser.add_argument("--dataset",          type=str,   required=True)
    parser.add_argument("--iter",             type=int,   default=300)
    parser.add_argument("--trial",            type=int,   default=50)
    parser.add_argument("--jobs",             type=int,   default=8)
    parser.add_argument("--nfold",            type=int,   default=3)
    parser.add_argument("--ncv",              type=int,   default=8)
    # training parameters
    parser.add_argument("--learning_rate",    type=float, default=0.05)
    parser.add_argument("--num_leaves",       type=int,   default=None)
    parser.add_argument("--random_seed",      type=int,   default=1)
    parser.add_argument("--max_depth",        type=int,   default=8)
    parser.add_argument("--min_child_samples",type=int,   default=None)
    parser.add_argument("--min_child_weight", type=float, default=None)
    parser.add_argument("--subsample",        type=float, default=1)
    parser.add_argument("--colsample_bytree", type=float, default=None)
    parser.add_argument("--colsample_bylevel",type=float, default=None)
    parser.add_argument("--colsample_bynode", type=float, default=None)
    parser.add_argument("--max_bin",          type=int,   default=256)
    parser.add_argument("--min_data_in_bin",  type=int,   default=None)
    parser.add_argument("--reg_alpha",        type=float, default=None)
    parser.add_argument("--reg_lambda",       type=float, default=None)
    parser.add_argument("--min_split_gain",   type=float, default=None)
    args = parser.parse_args()

    # base parameters
    params_const = {
        "learning_rate"    : args.learning_rate,
        "num_leaves"       : args.num_leaves,
        "is_gpu"           : False,
        "random_seed"      : args.random_seed,
        "max_depth"        : args.max_depth,
        "min_child_samples": args.min_child_samples,
        "min_child_weight" : args.min_child_weight,
        "subsample"        : args.subsample,
        "colsample_bytree" : args.colsample_bytree,
        "colsample_bylevel": args.colsample_bylevel,
        "colsample_bynode" : args.colsample_bynode,
        "max_bin"          : args.max_bin,
        "min_data_in_bin"  : args.min_data_in_bin,
        "reg_alpha"        : args.reg_alpha,
        "reg_lambda"       : args.reg_lambda,
        "min_split_gain"   : args.min_split_gain,
        "path_smooth"      : None,
        "verbosity"        : None,
        "is_softmax"       : None,
    }

    # dataset
    reg      = DatasetRegistry()
    filepath = "params.db"
    LOGGER.info(f"Tuning {args.dataset}...", color=["BOLD", "CYAN", "UNDERLINE"])
    dataset  = reg.create(args.dataset, seed=args.random_seed)
    assert dataset.metadata.supported_task == "multiclass"

    # Tuning
    train_x, train_y, valid_x, valid_y = dataset.load_data(format="numpy", split_type="test", test_size=0.3333)
    n_class = dataset.metadata.n_classes
    sampler = optuna.samplers.TPESampler()
    try: optuna.delete_study(study_name=f"{args.mode}_{args.dataset}", storage=f'sqlite:///{filepath}')
    except KeyError: pass
    study   = optuna.create_study(study_name=f"{args.mode}_{args.dataset}", storage=f'sqlite:///{filepath}', sampler=sampler, directions=["minimize"])
    func    = partial(tune_parameter,
        mode=args.mode, num_class=n_class, n_jobs=args.jobs, eval_string={
            'lgb': 'model.booster.best_score["valid_0"]["multi_logloss"]',
            'xgb': 'model.evals_result["valid_0"]["mlogloss"][model.booster.best_iteration]',
            'cat': 'model.evals_result["validation"]["MultiClass"][model.best_iteration]',
        }[args.mode],
        x_train=train_x, y_train=train_y, loss_func="multi", num_iterations=args.iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval="multi", sample_weight="balanced",
        early_stopping_rounds=20, early_stopping_idx=0,
        params_const = params_const,
        params_search='''{
            "min_child_weight" : trial.suggest_float("min_child_weight", 1e-4, 1e3, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 1e3, log=True),
            "min_split_gain"   : trial.suggest_float("min_split_gain", 1e-10, 1.0, log=True),
            "colsample_bynode" : trial.suggest_float("colsample_bynode", 0.1, 1.0, log=False),
            "subsample"        : trial.suggest_float("subsample", 0.1, 1.0, log=False),
        }'''
    )
    study.optimize(func, n_trials=args.trial)
    dictwk = {i: x.values[0] for i, x in enumerate(study.get_trials())}
    LOGGER.info(
        f"\nRESULT VALUES:\n{json.dumps({x:float(y) for x, y in dictwk.           items()}, indent=2)}\n" + 
        f"BEST PARAMETERS:\n{json.dumps({x:float(y) for x, y in study.best_params.items()}, indent=2)}",
        color=["BOLD", "GREEN"]
    )

    # Training
    cv1st = StratifiedKFold(n_splits=args.nfold, shuffle=True, random_state=args.random_seed)
    cv2nd = StratifiedKFold(n_splits=args.ncv,   shuffle=True, random_state=args.random_seed)
    data_x, data_y = dataset.load_data(format="numpy", split_type="train")
    list_eval = []
    for i_fold, (idx_cv, idx_test) in enumerate(cv1st.split(data_x, data_y)):
        cv_x,   cv_y    = data_x[idx_cv],   data_y[idx_cv]
        test_x, test_y  = data_x[idx_test], data_y[idx_test]
        list_prediction = []
        for i_cv, (idx_train, idx_valid) in enumerate(cv2nd.split(cv_x, cv_y)):
            train_x, train_y = cv_x[idx_train], cv_y[idx_train]
            valid_x, valid_y = cv_x[idx_valid], cv_y[idx_valid]
            model = KkGBDT(n_class, mode=args.mode, n_jobs=args.jobs, **(params_const | study.best_params))
            model.fit(
                train_x, train_y, loss_func="multi", num_iterations=args.iter,
                x_valid=valid_x, y_valid=valid_y, sample_weight="balanced",
                early_stopping_rounds=20, early_stopping_idx=0,
            )
            ndf_pred = model.predict(test_x, iteration_at=model.best_iteration)
            logloss  = log_loss(test_y, ndf_pred)
            list_prediction.append(ndf_pred)
            list_eval.append({
                "fold": i_fold, "cv": i_cv, "train": len(idx_train), "valid": len(idx_valid), 
                "test": len(idx_test), "is_ensemble": False, "eval": logloss,
            })
            LOGGER.info(f"fold={i_fold} cv={i_cv} train={len(idx_train)} valid={len(idx_valid)} test={len(idx_test)} eval={logloss}", color=["BOLD", "CYAN"])
        ndf_pred = np.stack(list_prediction).mean(axis=0)
        logloss  = log_loss(test_y, ndf_pred)
        list_eval.append({
            "fold": i_fold, "cv": None, "train": None, "valid": None, 
            "test": len(idx_test), "is_ensemble": True, "eval": logloss,
        })
        LOGGER.info(f"fold={i_fold} test={len(idx_test)} eval (ensemble score)={logloss}", color=["BOLD", "CYAN"])

    # Print results
    df = pd.DataFrame(list_eval)
    LOGGER.info(f"Mean: {df.loc[df['is_ensemble'] == True, 'eval'].mean()}", color=["BOLD", "GREEN"])
    LOGGER.info(f"\n{df.to_string(index=False)}")
