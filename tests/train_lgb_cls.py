import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from kktestdata import DatasetRegistry
from kkgbdt.model import KkGBDT
from kkgbdt.loss import CategoricalCrossEntropyLoss, CrossEntropyLoss, Accuracy, CategoricalFocalLoss, \
    CrossEntropyLossArgmax, BinaryCrossEntropyLoss, CrossEntropyNDCGLoss, LogitMarginL1Loss, BinaryCrossEntropyLoss
from kkgbdt.functions import log_loss
from kklogger import set_logger


np.random.seed(0)
LOGGER = set_logger(__name__)


if __name__ == "__main__":
    reg     = DatasetRegistry()
    dataset = reg.create("gas-drift")
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataset.load_data(
        format="numpy", split_type="valid", test_size=0.3, valid_size=0.2
    )
    n_class = dataset.metadata.n_classes
    n_iter  = 100
    lr      = 0.2
    max_bin = 64
    ndepth  = -1
    valeval = {}
    LOGGER.info("data train", color=["BOLD", "CYAN"])
    LOGGER.info(f"\n{pd.DataFrame(train_y).groupby(0).size()}")
    LOGGER.info("data valid", color=["BOLD", "CYAN"])
    LOGGER.info(f"\n{pd.DataFrame(valid_y).groupby(0).size()}")

    LOGGER.info("Test training stop", color=["BOLD", "UNDERLINE", "GREEN"])
    model   = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func="multi", num_iterations=10, sample_weight=["balanced", np.random.rand(train_x.shape[0])],
        train_stopping_val=0.1, train_stopping_rounds=5, train_stopping_is_over=True
    )
    model.fit(
        train_x, train_y, loss_func="multi", num_iterations=10, sample_weight=["balanced", np.random.rand(train_x.shape[0])],
        train_stopping_time=0.01, train_stopping_rounds=5
    )

    LOGGER.info("public loss binary", color=["BOLD", "UNDERLINE", "GREEN"])
    model   = KkGBDT(1, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, (train_y == 1).astype(int), loss_func="binary", num_iterations=n_iter,
        x_valid=valid_x, y_valid=(valid_y == 1).astype(int), sample_weight="balanced",
        early_stopping_rounds=20, early_stopping_idx=0,
    )
    ndf_pred = model.predict(test_x, iteration_at=model.best_iteration)
    assert model.best_iteration < n_iter
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    valeval["binary_log"] = BinaryCrossEntropyLoss()(ndf_pred, (test_y == 1).astype(int))
    valeval["binary_acc"] = Accuracy(top_k=1)(ndf_pred, (test_y == 1).astype(int))

    LOGGER.info("custom public loss binary", color=["BOLD", "UNDERLINE", "GREEN"])
    model   = KkGBDT(1, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, (train_y == 1).astype(int), loss_func=BinaryCrossEntropyLoss(), num_iterations=n_iter,
        x_valid=valid_x, y_valid=(valid_y == 1).astype(int), sample_weight="balanced",
        early_stopping_rounds=20, early_stopping_idx=0,
    )
    ndf_pred = model.predict(test_x, iteration_at=model.best_iteration)
    assert model.best_iteration < n_iter
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    assert model.loss == KkGBDT.from_dict(model.to_dict()).loss
    valeval["BinaryCrossEntropyLoss_log"] = BinaryCrossEntropyLoss()(ndf_pred, (test_y == 1).astype(int))
    valeval["BinaryCrossEntropyLoss_acc"] = Accuracy(top_k=1)(ndf_pred, (test_y == 1).astype(int))

    LOGGER.info("public loss multiclass ( no validation )", color=["BOLD", "UNDERLINE", "GREEN"])
    model   = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func="multi", num_iterations=n_iter, sample_weight="balanced",
    )
    ndf_pred = model.predict(test_x, iteration_at=model.best_iteration)
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    valeval["multiclass_log"] = log_loss(test_y, ndf_pred)
    valeval["multiclass_acc"] = Accuracy(top_k=2)(ndf_pred, test_y)

    LOGGER.info("public loss multiclass", color=["BOLD", "UNDERLINE", "GREEN"])
    model   = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func="multi", num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["multi", Accuracy(top_k=2)], sample_weight="balanced",
        early_stopping_rounds=20, early_stopping_idx=0,
    )
    ndf_pred = model.predict(test_x, iteration_at=model.best_iteration)
    assert model.best_iteration < n_iter
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    valeval["multiclass_log"] = log_loss(test_y, ndf_pred)
    valeval["multiclass_acc"] = Accuracy(top_k=2)(ndf_pred, test_y)

    LOGGER.info('public loss with "use_quantized_grad"', color=["BOLD", "UNDERLINE", "GREEN"])
    try:
        model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth, use_quantized_grad=True, num_grad_quant_bins=4, )
        LOGGER.info("train without validation", color=["BOLD", "CYAN"])
        model.fit(train_x, train_y, loss_func="multi", num_iterations=10)
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
    model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func=CategoricalCrossEntropyLoss(n_class, smoothing=1e-4), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["__copy__", CategoricalCrossEntropyLoss(n_class), Accuracy(top_k=2)],
        early_stopping_rounds=20, early_stopping_idx=0, sample_weight="balanced",
    )
    ndf_pred = model.predict(test_x)
    assert model.best_iteration < n_iter
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    assert model.loss == KkGBDT.from_dict(model.to_dict()).loss
    valeval["CategoryCE_log"] = log_loss(test_y, ndf_pred)
    valeval["CategoryCE_acc"] = Accuracy(top_k=2)(ndf_pred, test_y)

    LOGGER.info("custom loss FocalLoss", color=["BOLD", "UNDERLINE", "GREEN"])
    model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func=CategoricalFocalLoss(n_class, gamma=0.5, dx=1e-5), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["__copy__", CategoricalCrossEntropyLoss(n_class), Accuracy(top_k=2)],
        early_stopping_rounds=20, early_stopping_idx=0, sample_weight="balanced",
    )
    ndf_pred = model.predict(test_x)
    assert model.best_iteration < n_iter
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    assert model.loss == KkGBDT.from_dict(model.to_dict()).loss
    valeval["FocalLoss_log"] = log_loss(test_y, ndf_pred)
    valeval["FocalLoss_acc"] = Accuracy(top_k=2)(ndf_pred, test_y)

    LOGGER.info("custom loss LogitMarginL1Loss", color=["BOLD", "UNDERLINE", "GREEN"])
    model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func=LogitMarginL1Loss(n_class, margin=20.0, alpha=0.01, dx=1e-5), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["__copy__", CategoricalCrossEntropyLoss(n_class), Accuracy(top_k=2)],
        early_stopping_rounds=20, early_stopping_idx=0, sample_weight="balanced",
    )
    ndf_pred = model.predict(test_x)
    assert model.best_iteration < n_iter
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    assert model.loss == KkGBDT.from_dict(model.to_dict()).loss
    valeval["LogitMarginL1Loss_log"] = log_loss(test_y, ndf_pred)
    valeval["LogitMarginL1Loss_acc"] = Accuracy(top_k=2)(ndf_pred, test_y)

    LOGGER.info("custom loss CrossEntropyLoss", color=["BOLD", "UNDERLINE", "GREEN"])
    model      = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    train_y_ce = np.zeros((train_y.shape[0], n_class), dtype=int)
    train_y_ce[np.arange(train_y_ce.shape[0]), train_y] = 1
    valid_y_ce = np.zeros((valid_y.shape[0], n_class), dtype=int)
    valid_y_ce[np.arange(valid_y_ce.shape[0]), valid_y] = 1
    model.fit(
        train_x, train_y_ce, loss_func=CrossEntropyLoss(n_class, dx=1e-5), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y_ce, loss_func_eval=["__copy__", CrossEntropyLossArgmax(n_class), Accuracy(top_k=2)],
        early_stopping_rounds=20, early_stopping_idx=0
    )
    ndf_pred = model.predict(test_x)
    assert model.best_iteration < n_iter
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    assert model.loss == KkGBDT.from_dict(model.to_dict()).loss
    valeval["CrossEntropyLoss_log"] = log_loss(test_y, ndf_pred)
    valeval["CrossEntropyLoss_acc"] = Accuracy(top_k=2)(ndf_pred, test_y)

    LOGGER.info("custom loss CrossEntropyNDCGLoss", color=["BOLD", "UNDERLINE", "GREEN"])
    train_y_ndcg = np.ones((train_y.shape[0], n_class), dtype=int)
    train_y_ndcg[np.arange(train_y_ndcg.shape[0]), train_y] = 20 # score
    valid_y_ndcg = np.ones((valid_y.shape[0], n_class), dtype=int)
    valid_y_ndcg[np.arange(valid_y_ndcg.shape[0]), valid_y] = 20
    model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y_ndcg, loss_func=CrossEntropyNDCGLoss(n_class), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y_ndcg, loss_func_eval="__copy__",
        early_stopping_rounds=20, early_stopping_idx=0, sample_weight=None,
    )
    ndf_pred = model.predict(test_x)
    assert model.best_iteration < n_iter
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    assert model.loss == KkGBDT.from_dict(model.to_dict()).loss
    valeval["CrossEntropyNDCGLoss_log"] = log_loss(test_y, ndf_pred)
    valeval["CrossEntropyNDCGLoss_acc"] = Accuracy(top_k=2)(ndf_pred, test_y)

    LOGGER.info(f"{json.dumps({x:float(y) for x, y in valeval.items()}, indent=2)}")
