import numpy as np
import pandas as pd
from kktestdata import DatasetRegistry
from kkgbdt.model import KkGBDT
from kkgbdt.loss import CategoricalCrossEntropyLoss, Accuracy, BinaryCrossEntropyLoss, CategoricalFocalLoss
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
    lr      = 1.0
    max_bin = 64
    ndepth  = 7
    valeval = {}
    LOGGER.info("data train", color=["BOLD", "CYAN"])
    LOGGER.info(f"\n{pd.DataFrame(train_y).groupby(0).size()}")
    LOGGER.info("data valid", color=["BOLD", "CYAN"])
    LOGGER.info(f"\n{pd.DataFrame(valid_y).groupby(0).size()}")

    LOGGER.info("public loss binary", color=["BOLD", "UNDERLINE", "GREEN"])
    model   = KkGBDT(1, mode="cat", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, (train_y == 1).astype(int), loss_func="binary", num_iterations=n_iter,
        x_valid=valid_x, y_valid=(valid_y == 1).astype(int), sample_weight="balanced",
        early_stopping_rounds=20, early_stopping_idx=0,
    )
    ndf_pred = model.predict(test_x, iteration_at=model.best_iteration)
    assert model.best_iteration < (n_iter - 1)
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    valeval["binary_log"] = BinaryCrossEntropyLoss()(ndf_pred, (test_y == 1).astype(int))
    valeval["binary_acc"] = Accuracy(top_k=1)(ndf_pred, (test_y == 1).astype(int))

    LOGGER.info("custom public loss binary", color=["BOLD", "UNDERLINE", "GREEN"])
    model   = KkGBDT(2, mode="cat", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, (train_y == 1).astype(int), loss_func=BinaryCrossEntropyLoss(), num_iterations=n_iter,
        x_valid=valid_x, y_valid=(valid_y == 1).astype(int), sample_weight="balanced",
        early_stopping_rounds=20, early_stopping_idx=0,
    )
    ndf_pred = model.predict(test_x, iteration_at=model.best_iteration)
    assert model.is_softmax == True
    assert model.best_iteration < (n_iter - 1)
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    assert model.loss == KkGBDT.from_dict(model.to_dict()).loss
    valeval["BinaryCrossEntropyLoss_log"] = BinaryCrossEntropyLoss()(ndf_pred, (test_y == 1).astype(int))
    valeval["BinaryCrossEntropyLoss_acc"] = Accuracy(top_k=1)(ndf_pred, (test_y == 1).astype(int))

    LOGGER.info("public loss multiclass ( no validation )", color=["BOLD", "UNDERLINE", "GREEN"])
    model   = KkGBDT(n_class, mode="cat", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func="multi", num_iterations=n_iter, sample_weight="balanced",
    )
    ndf_pred = model.predict(test_x, iteration_at=model.best_iteration)
    assert model.is_softmax == True
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    valeval["multiclass_log"] = log_loss(test_y, ndf_pred)
    valeval["multiclass_acc"] = Accuracy(top_k=2)(ndf_pred, test_y)

    LOGGER.info("public loss multiclass", color=["BOLD", "UNDERLINE", "GREEN"])
    model   = KkGBDT(n_class, mode="cat", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func="multi", num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["multi", "acc"], sample_weight="balanced",
        early_stopping_rounds=20, early_stopping_idx=0,
    )
    ndf_pred = model.predict(test_x, iteration_at=model.best_iteration)
    assert model.is_softmax == True
    assert model.best_iteration < (n_iter - 1)
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    valeval["multiclass_log"] = log_loss(test_y, ndf_pred)
    valeval["multiclass_acc"] = Accuracy(top_k=2)(ndf_pred, test_y)

    n_iter = 10

    LOGGER.info("custom loss CategoryCE", color=["BOLD", "UNDERLINE", "GREEN"])
    model = KkGBDT(n_class, mode="cat", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func=CategoricalCrossEntropyLoss(n_class, smoothing=1e-4), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=[CategoricalCrossEntropyLoss(n_class), "multi", "acc"],
        early_stopping_rounds=20, early_stopping_idx=0, sample_weight="balanced",
    )
    ndf_pred = model.predict(test_x)
    assert model.is_softmax == True
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    assert model.loss == KkGBDT.from_dict(model.to_dict()).loss
    valeval["CategoryCE_log"] = log_loss(test_y, ndf_pred)
    valeval["CategoryCE_acc"] = Accuracy(top_k=2)(ndf_pred, test_y)

    LOGGER.info("custom loss FocalLoss", color=["BOLD", "UNDERLINE", "GREEN"])
    model = KkGBDT(n_class, mode="cat", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func=CategoricalFocalLoss(n_class, gamma=0.5, dx=1e-5), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["__copy__", "multi", "acc"],
        early_stopping_rounds=20, early_stopping_idx=0, sample_weight="balanced",
    )
    ndf_pred = model.predict(test_x)
    assert model.is_softmax == True
    assert np.all(ndf_pred == KkGBDT.from_dict(model.to_dict()).predict(test_x))
    assert model.loss == KkGBDT.from_dict(model.to_dict()).loss
    valeval["FocalLoss_log"] = log_loss(test_y, ndf_pred)
    valeval["FocalLoss_acc"] = Accuracy(top_k=2)(ndf_pred, test_y)
