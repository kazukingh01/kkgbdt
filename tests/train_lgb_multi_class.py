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
    model   = KkGBDT(n_class, mode="lgb", max_bin=64)
    model.fit(
        train_x, train_y, loss_func="multiclass", num_iterations=100,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["multiclass"], sample_weight=["balanced", np.random.rand(train_x.shape[0])],
    )
    print(model.predict(valid_x, is_softmax=False))
    print(f"best iteration the model.booster have is: {model.booster.best_iteration}")
    print(log_loss(valid_y, model.predict(valid_x, is_softmax=False, iteration_at=model.best_iteration)))
    # public loss with "use_quantized_grad"
    model = KkGBDT(n_class, mode="lgb", max_bin=64, use_quantized_grad=True, num_grad_quant_bins=8, )
    model.fit(
        train_x, train_y, loss_func="multiclass", num_iterations=100,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["multiclass"], sample_weight="balanced",
    )
    print(model.predict(valid_x, is_softmax=False))
    print(f"best iteration the model.booster have is: {model.booster.best_iteration}")
    print(log_loss(valid_y, model.predict(valid_x, is_softmax=False, iteration_at=model.best_iteration)))
    # custom loss CategoryCE
    model = KkGBDT(n_class, mode="lgb", max_bin=64)
    model.fit(
        train_x, train_y, loss_func=CategoricalCrossEntropyLoss(n_class), num_iterations=100,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=[CategoricalCrossEntropyLoss(n_class), Accuracy(top_k=n_class//2)],
        early_stopping_rounds=20, early_stopping_name=0, sample_weight="balanced",
    )
    print(model.predict(valid_x, is_softmax=True))
    # custom loss CE
    model      = KkGBDT(n_class, mode="lgb", max_bin=64)
    train_y_ce = np.zeros((train_y.shape[0], n_class))
    train_y_ce[np.arange(train_y_ce.shape[0]), train_y] = 1
    valid_y_ce = np.zeros((valid_y.shape[0], n_class))
    valid_y_ce[np.arange(valid_y_ce.shape[0]), valid_y] = 1
    model.fit(
        train_x, train_y_ce.astype(int), loss_func=CrossEntropyLoss(n_class), num_iterations=100,
        x_valid=valid_x, y_valid=valid_y_ce.astype(int), loss_func_eval=[CrossEntropyLoss(n_class), Accuracy(top_k=n_class//2)],
        early_stopping_rounds=None, early_stopping_name=0, 
        stopping_name="ce", stopping_val=3.70, stopping_rounds=5, stopping_is_over=True
    )
    print(model.predict(valid_x, is_softmax=True))
    print(f"best iteration the model.booster have is: {model.booster.best_iteration}")
    print(CrossEntropyLoss(n_class)(model.predict(valid_x, is_softmax=False, iteration_at=model.best_iteration), valid_y_ce))
