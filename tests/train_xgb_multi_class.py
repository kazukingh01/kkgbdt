import numpy as np
from sklearn.datasets import fetch_covtype 
from kkgbdt.model import KkGBDT
from kkgbdt.loss import CategoricalCrossEntropyLoss, CrossEntropyLoss, Accuracy
np.random.seed(0)


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
    # public loss
    model   = KkGBDT(n_class, mode="xgb", max_bin=64)
    model.fit(
        train_x, train_y, loss_func="multi:softmax", num_iterations=100,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["mlogloss"], sample_weight="balanced"
    )
    print(model.predict(valid_x, is_softmax=False))
    print(log_loss(valid_y, model.predict(valid_x, is_softmax=True)))
    # custom loss CategoryCE
    model = KkGBDT(n_class, mode="xgb", max_bin=64)
    model.fit(
        train_x, train_y, loss_func=CategoricalCrossEntropyLoss(n_class), num_iterations=100,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=[CategoricalCrossEntropyLoss(n_class), "mlogloss"],
        early_stopping_rounds=20, early_stopping_name=0, sample_weight="balanced",
    )
    print(model.predict(valid_x, is_softmax=True, iteration_at=model.best_iteration))
    print(log_loss(valid_y, model.predict(valid_x, is_softmax=True)))
    # binary logloss with multi_strategy
    model      = KkGBDT(1, mode="xgb", max_bin=64, multi_strategy="multi_output_tree")
    train_y_ce = np.zeros((train_y.shape[0], n_class))
    train_y_ce[np.arange(train_y_ce.shape[0]), train_y] = 1
    valid_y_ce = np.zeros((valid_y.shape[0], n_class))
    valid_y_ce[np.arange(valid_y_ce.shape[0]), valid_y] = 1
    model.fit(
        train_x, train_y_ce.astype(int), loss_func="binary:logistic", num_iterations=100,
        x_valid=valid_x, y_valid=valid_y_ce.astype(int), loss_func_eval=["logloss"],
        early_stopping_rounds=None
    )
    print(model.predict(valid_x, is_softmax=True))
    print(CrossEntropyLoss(n_class)(model.predict(valid_x, is_softmax=False), valid_y_ce))
    # custom loss CE with multi_strategy
    model = KkGBDT(1, mode="xgb", max_bin=64, multi_strategy="multi_output_tree")
    model.fit(
        train_x, train_y_ce.astype(int), loss_func=CrossEntropyLoss(n_class), num_iterations=100,
        x_valid=valid_x, y_valid=valid_y_ce.astype(int), loss_func_eval=[CrossEntropyLoss(n_class), "logloss"],
        early_stopping_rounds=None
    )
    print(model.predict(valid_x, is_softmax=True))
    print(CrossEntropyLoss(n_class)(model.predict(valid_x, is_softmax=False, iteration_at=model.best_iteration), valid_y_ce))
