import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype 
from kkgbdt.model import KkGBDT, set_all_loglevel
from kkgbdt.loss import CategoricalCrossEntropyLoss, CrossEntropyLoss, Accuracy, FocalLoss, \
    CrossEntropyLossArgmax, BinaryCrossEntropyLoss, CrossEntropyNDCGLoss, LogitMarginL1Loss
from kklogger import set_logger


np.random.seed(0)
LOGGER = set_logger(__name__)


def log_loss(y: np.ndarray, x: np.ndarray):
    assert isinstance(y, np.ndarray) and len(y.shape) == 1
    assert isinstance(x, np.ndarray) and len(x.shape) == 2
    assert y.dtype in [int, np.int8, np.int16, np.int32, np.int64]
    ndf = x[np.arange(x.shape[0]), y]
    return (-1 * np.log(ndf)).mean()


if __name__ == "__main__":
    data = fetch_covtype()
    train_x = data["data"  ][:-data["target"].shape[0]//5 ]
    train_y = data["target"][:-data["target"].shape[0]//5 ] - 1
    valid_x = data["data"  ][ -data["target"].shape[0]//5:]
    valid_y = data["target"][ -data["target"].shape[0]//5:] - 1
    n_class = np.unique(train_y).shape[0]
    n_iter  = 500
    lr      = 0.05
    max_bin = 64
    valeval = {}
    LOGGER.info("data train", color=["BOLD", "CYAN"])
    LOGGER.info(f"\n{pd.DataFrame(train_y).groupby(0).size()}")
    LOGGER.info("data valid", color=["BOLD", "CYAN"])
    LOGGER.info(f"\n{pd.DataFrame(valid_y).groupby(0).size()}")

    LOGGER.info("Test public loss", color=["BOLD", "UNDERLINE", "GREEN"])
    model   = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin)
    LOGGER.info("train without validation", color=["BOLD", "CYAN"])
    model.fit(train_x, train_y, loss_func="multiclass", num_iterations=10)
    LOGGER.info("train with validation", color=["BOLD", "CYAN"])
    model.fit(
        train_x, train_y, loss_func="multiclass", num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["multiclass", Accuracy(top_k=2)], sample_weight=["balanced", np.random.rand(train_x.shape[0])],
        early_stopping_rounds=20, early_stopping_name=0,
    )
    ndf_pred = model.predict(valid_x, iteration_at=model.best_iteration)
    valeval["multiclass_log"] = log_loss(valid_y, ndf_pred)
    valeval["multiclass_acc"] = Accuracy(top_k=2)(ndf_pred, valid_y)

    LOGGER.info('public loss with "use_quantized_grad"', color=["BOLD", "UNDERLINE", "GREEN"])
    try:
        model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, use_quantized_grad=True, num_grad_quant_bins=4, )
        LOGGER.info("train without validation", color=["BOLD", "CYAN"])
        model.fit(train_x, train_y, loss_func="multiclass", num_iterations=10)
    except lgb.basic.LightGBMError:
        """
        I've got this error but this is not reproducible. If I do many times, the training will be succeeded a few times.
            Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            File "/home/kkazuki/10.git/kkgbdt/kkgbdt/model.py", line 129, in fit
                self.booster = self.train_func(
                            ^^^^^^^^^^^^^^^^
            File "/home/kkazuki/10.git/kkgbdt/kkgbdt/model.py", line 429, in train_lgb
                model = lgb.train(
                        ^^^^^^^^^^
            File "/home/kkazuki/10.git/kkgbdt/venv/lib/python3.12/site-packages/lightgbm/engine.py", line 307, in train
                booster.update(fobj=fobj)
            File "/home/kkazuki/10.git/kkgbdt/venv/lib/python3.12/site-packages/lightgbm/basic.py", line 4135, in update
                _safe_call(
            File "/home/kkazuki/10.git/kkgbdt/venv/lib/python3.12/site-packages/lightgbm/basic.py", line 296, in _safe_call
                raise LightGBMError(_LIB.LGBM_GetLastError().decode("utf-8"))
            lightgbm.basic.LightGBMError: Check failed: (best_split_info.left_count) > (0) at /__w/1/s/lightgbm-python/src/treelearner/serial_tree_learner.cpp, line 868 .
        """
        LOGGER.warning(f"lightgbm.basic.LightGBMError happened. If you don't want to use use_quantized_grad, ignore this error.")

    LOGGER.info("custom loss CategoryCE", color=["BOLD", "UNDERLINE", "GREEN"])
    model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin)
    model.fit(
        train_x, train_y, loss_func=CategoricalCrossEntropyLoss(n_class, smoothing=1e-4), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["__copy__", CategoricalCrossEntropyLoss(n_class), Accuracy(top_k=2)],
        early_stopping_rounds=20, early_stopping_name=0, sample_weight="balanced",
    )
    ndf_pred = model.predict(valid_x)
    valeval["CategoryCE_log"] = log_loss(valid_y, ndf_pred)
    valeval["CategoryCE_acc"] = Accuracy(top_k=2)(ndf_pred, valid_y)

    LOGGER.info("custom loss FocalLoss", color=["BOLD", "UNDERLINE", "GREEN"])
    model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin)
    model.fit(
        train_x, train_y, loss_func=FocalLoss(n_class, gamma=0.5), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["__copy__", CategoricalCrossEntropyLoss(n_class), Accuracy(top_k=2)],
        early_stopping_rounds=20, early_stopping_name=0, sample_weight="balanced",
    )
    ndf_pred = model.predict(valid_x)
    valeval["FocalLoss_log"] = log_loss(valid_y, ndf_pred)
    valeval["FocalLoss_acc"] = Accuracy(top_k=2)(ndf_pred, valid_y)

    LOGGER.info("custom loss LogitMarginL1Loss", color=["BOLD", "UNDERLINE", "GREEN"])
    model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin)
    model.fit(
        train_x, train_y, loss_func=LogitMarginL1Loss(n_class, margin=2.0, alpha=0.01), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["__copy__", CategoricalCrossEntropyLoss(n_class), Accuracy(top_k=2)],
        early_stopping_rounds=20, early_stopping_name=0, sample_weight="balanced",
    )
    ndf_pred = model.predict(valid_x)
    valeval["LogitMarginL1Loss_log"] = log_loss(valid_y, ndf_pred)
    valeval["LogitMarginL1Loss_acc"] = Accuracy(top_k=2)(ndf_pred, valid_y)

    LOGGER.info("custom loss CrossEntropyLoss", color=["BOLD", "UNDERLINE", "GREEN"])
    model      = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin)
    train_y_ce = np.zeros((train_y.shape[0], n_class), dtype=int)
    train_y_ce[np.arange(train_y_ce.shape[0]), train_y] = 1
    valid_y_ce = np.zeros((valid_y.shape[0], n_class), dtype=int)
    valid_y_ce[np.arange(valid_y_ce.shape[0]), valid_y] = 1
    model.fit(
        train_x, train_y_ce, loss_func=CrossEntropyLoss(n_class), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y_ce, loss_func_eval=["__copy__", CrossEntropyLossArgmax(n_class), Accuracy(top_k=2)],
        early_stopping_rounds=20, early_stopping_name=0, 
        stopping_name="ce", stopping_val=3.70, stopping_rounds=5, stopping_is_over=True
    )
    ndf_pred = model.predict(valid_x)
    valeval["CrossEntropyLoss_log"] = log_loss(valid_y, ndf_pred)
    valeval["CrossEntropyLoss_acc"] = Accuracy(top_k=2)(ndf_pred, valid_y)

    LOGGER.info("custom loss BinaryCrossEntropyLoss", color=["BOLD", "UNDERLINE", "GREEN"])
    list_pred = []
    for i, (_train_y, _valid_y) in enumerate(zip(train_y_ce.T, valid_y_ce.T)):
        model = KkGBDT(1, mode="lgb", learning_rate=lr, max_bin=max_bin)
        model.fit(
            train_x, _train_y, loss_func=BinaryCrossEntropyLoss(), num_iterations=n_iter,
            x_valid=valid_x, y_valid=_valid_y, loss_func_eval=["__copy__", Accuracy(top_k=1)],
            early_stopping_rounds=20, early_stopping_name=0, sample_weight="balanced",
        )
        list_pred.append(model.predict(valid_x))
    ndf = np.stack(list_pred).T
    ndf_pred = ndf / ndf.sum(axis=-1).reshape(-1, 1)
    valeval["BinaryCrossEntropyLoss_log"] = log_loss(valid_y, ndf_pred)
    valeval["BinaryCrossEntropyLoss_acc"] = Accuracy(top_k=2)(ndf_pred, valid_y)

    LOGGER.info("custom loss CrossEntropyNDCGLoss", color=["BOLD", "UNDERLINE", "GREEN"])
    train_y_ndcg = np.ones((train_y.shape[0], n_class), dtype=int)
    train_y_ndcg[np.arange(train_y_ndcg.shape[0]), train_y] = 20 # score
    valid_y_ndcg = np.ones((valid_y.shape[0], n_class), dtype=int)
    valid_y_ndcg[np.arange(valid_y_ndcg.shape[0]), valid_y] = 20
    model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin)
    model.fit(
        train_x, train_y_ndcg, loss_func=CrossEntropyNDCGLoss(n_class), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y_ndcg, loss_func_eval="__copy__",
        early_stopping_rounds=20, early_stopping_name=0, sample_weight=None,
    )
    ndf_pred = model.predict(valid_x)
    valeval["CrossEntropyNDCGLoss_log"] = log_loss(valid_y, ndf_pred)
    valeval["CrossEntropyNDCGLoss_acc"] = Accuracy(top_k=2)(ndf_pred, valid_y)

    LOGGER.info(f"{json.dumps(valeval, indent=2)}")
