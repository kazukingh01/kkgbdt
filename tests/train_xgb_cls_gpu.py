import numpy as np
from sklearn.datasets import fetch_covtype 
from kkgbdt.model import KkGBDT
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
    model   = KkGBDT(n_class, mode="xgb", max_bin=64, is_gpu=True)
    model.fit(
        train_x, train_y, loss_func="multi:softmax", num_iterations=100,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["mlogloss"], sample_weight="balanced"
    )
    print(model.predict(valid_x, is_softmax=False))
    print(log_loss(valid_y, model.predict(valid_x, is_softmax=True)))
