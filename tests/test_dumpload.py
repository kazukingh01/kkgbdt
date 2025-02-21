import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype 
from kkgbdt.model import KkGBDT
from kkgbdt.loss import CategoricalCrossEntropyLoss, Accuracy
from kklogger import set_logger


np.random.seed(0)
LOGGER = set_logger(__name__)


if __name__ == "__main__":
    data = fetch_covtype()
    train_x = data["data"  ][:-data["target"].shape[0]//5 ]
    train_y = data["target"][:-data["target"].shape[0]//5 ] - 1
    valid_x = data["data"  ][ -data["target"].shape[0]//5:]
    valid_y = data["target"][ -data["target"].shape[0]//5:] - 1
    n_class = np.unique(train_y).shape[0]
    n_iter  = 200
    lr      = 0.3
    max_bin = 64
    ndepth  = -1

    # lgb normal loss
    model   = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func="multiclass", num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["multiclass", Accuracy(top_k=2)], sample_weight="balanced",
        early_stopping_rounds=20, early_stopping_name=0,
    )
    ins       = KkGBDT.load(model.dump())
    ndf_pred1 = model.predict(valid_x)
    ndf_pred2 = ins.predict(valid_x)
    assert np.allclose(ndf_pred1, ndf_pred2)
    assert model.to_dict() == ins.to_dict()

    # lgb custom loss
    model = KkGBDT(n_class, mode="lgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func=CategoricalCrossEntropyLoss(n_class, smoothing=1e-4), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["__copy__", CategoricalCrossEntropyLoss(n_class), Accuracy(top_k=2)],
        early_stopping_rounds=20, early_stopping_name=0, sample_weight="balanced",
    )
    ins       = KkGBDT.load(model.dump())
    ndf_pred1 = model.predict(valid_x)
    ndf_pred2 = ins.predict(valid_x)
    assert np.allclose(ndf_pred1, ndf_pred2)
    assert model.to_dict() == ins.to_dict()

    # xgb normal loss
    model   = KkGBDT(n_class, mode="xgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func="multi:softmax", num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["mlogloss", Accuracy(top_k=2)], sample_weight="balanced",
        early_stopping_rounds=20, early_stopping_name=0,
    )
    ins       = KkGBDT.load(model.dump())
    ndf_pred1 = model.predict(valid_x, is_softmax=True)
    ndf_pred2 = ins.predict(valid_x, is_softmax=True)
    assert np.allclose(ndf_pred1, ndf_pred2)
    assert model.to_dict() == ins.to_dict()

    # xgb custom loss
    model = KkGBDT(n_class, mode="xgb", learning_rate=lr, max_bin=max_bin, max_depth=ndepth)
    model.fit(
        train_x, train_y, loss_func=CategoricalCrossEntropyLoss(n_class, smoothing=1e-4), num_iterations=n_iter,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["__copy__", CategoricalCrossEntropyLoss(n_class), Accuracy(top_k=2)],
        early_stopping_rounds=20, early_stopping_name=0, sample_weight="balanced",
    )
    ndf_pred = model.predict(valid_x)
    ins       = KkGBDT.load(model.dump())
    ndf_pred1 = model.predict(valid_x)
    ndf_pred2 = ins.predict(valid_x)
    assert np.allclose(ndf_pred1, ndf_pred2)
    assert model.to_dict() == ins.to_dict()
