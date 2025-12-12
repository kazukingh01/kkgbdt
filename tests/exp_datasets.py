import argparse, datetime
from kktestdata import DatasetRegistry
from kkgbdt.model import KkGBDT
from kkgbdt.functions import log_loss
from kklogger import set_logger


LOGGER = set_logger(__name__)
LIST_DATASEET = [
    "Fashion-MNIST","KDDCup99","SDSS17","covertype","gas-drift","helena","hiva_agnostic","mnist_784","nursery",
    "shuttle","splice","students_dropout_and_academic_success","tamilnadu-electricity","walking-activity"
]
PARAMS_CONST = {
    "learning_rate"    : 0.05,
    "num_leaves"       : None,
    "is_gpu"           : False,
    "random_seed"      : 0,
    "max_depth"        : 8,
    "min_child_samples": None,
    "min_child_weight" : float("inf"),
    "subsample"        : 1,
    "colsample_bytree" : None,
    "colsample_bylevel": float("inf"),
    "colsample_bynode" : float("inf"),
    "max_bin"          : 128,
    "min_data_in_bin"  : None,
    "reg_alpha"        : None,
    "reg_lambda"       : float("inf"),
    "min_split_gain"   : None,
    "path_smooth"      : None,
    "verbosity"        : None,
}
PATAMS_CONST_LGB = (PARAMS_CONST | {
    "num_leaves"       : (2 ** PARAMS_CONST["max_depth"]), # depth = 8
    "max_depth"        : -1,
    "min_child_weight" : None,
    "colsample_bytree" : 1,
    "colsample_bylevel": 1,
    "colsample_bynode" : 1,
    "reg_lambda"       : None,
})
PATAMS_CONST_XGB = (PARAMS_CONST | {
    "min_child_weight" : None,
    "colsample_bytree" : 1,
    "colsample_bylevel": 1,
    "colsample_bynode" : 1,
    "reg_lambda"       : None,
})
PATAMS_CONST_CAT = (PARAMS_CONST | {
    "learning_rate"    : (PARAMS_CONST["learning_rate"] * 5.0),
    "min_child_weight" : None,
    "colsample_bytree" : 1, 
    "colsample_bylevel": 1,
    "colsample_bynode" : 1,
    "reg_lambda"       : None,
})
PARAMS_CONST_MODE = {
    "lgb": PATAMS_CONST_LGB,
    "xgb": PATAMS_CONST_XGB,
    "cat": PATAMS_CONST_CAT,
}


if __name__ == "__main__":
    import wandb
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter",    type=int, default=1000)
    parser.add_argument("--jobs",    type=int, default=8)
    parser.add_argument("--dataset", type=lambda x: [int(y.strip()) for y in x.split(",")], default=",".join([str(i) for i in range(len(LIST_DATASEET))]))
    parser.add_argument("--entity",  type=str,  required=True)
    parser.add_argument("--project", type=str, required=True)
    args = parser.parse_args()
    assert all(LIST_DATASEET[i] for i in args.dataset)

    now = datetime.datetime.now()
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        config={
            "datasets": LIST_DATASEET,
            "i_datasets": args.dataset,
            "params_const": PARAMS_CONST,
            "params_const_lgb": PATAMS_CONST_LGB,
            "params_const_xgb": PATAMS_CONST_XGB,
            "params_const_cat": PATAMS_CONST_CAT,
            "iters": args.iter,
            "jobs": args.jobs,
        },
    )

    reg = DatasetRegistry()
    for dataset_name in [LIST_DATASEET[i] for i in args.dataset]:
        LOGGER.info(f"Dataset: {dataset_name}...", color=["BOLD", "CYAN", "UNDERLINE"])
        for seed in range(1, 6):
            dataset = reg.create(dataset_name, seed=seed)
            train_x, train_y, valid_x, valid_y, test_x, test_y = dataset.load_data(
                format="numpy", split_type="valid", test_size=0.3, valid_size=0.2, 
            )
            if seed == 1:
                LOGGER.info(f"{dataset.to_display()}")
            n_class = dataset.metadata.n_classes
            for mode in ["lgb", "xgb", "cat"]:
                LOGGER.info(f"{mode} ( Default Params )", color=["BOLD", "GREEN"])
                model = KkGBDT(n_class, mode=mode, n_jobs=args.jobs, **(PARAMS_CONST | PARAMS_CONST_MODE[mode]))
                model.fit(
                    train_x, train_y, loss_func="multi", num_iterations=args.iter,
                    x_valid=valid_x, y_valid=valid_y, sample_weight="balanced",
                    early_stopping_rounds=20, early_stopping_idx=0,
                )
                ndf_pred = model.predict(test_x, iteration_at=model.best_iteration)
                logloss  = log_loss(test_y, ndf_pred)
                run.log({
                    "dataset": dataset_name, "mode": mode, "seed": seed, "logloss": logloss, "params": "init",
                    "best_iteration": model.best_iteration, "total_iteration": model.total_iteration,
                    "time_train": model.time_train, "time_iter": model.time_train / model.total_iteration,
                    "datetime": datetime.datetime.now(),
                })

    wandb.finish()



