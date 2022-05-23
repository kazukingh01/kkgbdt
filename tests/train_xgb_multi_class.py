import numpy as np
from kkgbdt.model import KkGBDT
from kkgbdt.loss import CategoricalCrossEntropyLoss, CrossEntropyLoss


if __name__ == "__main__":
    n_class, n_data = 5, 1000
    train_x = np.random.rand(n_data, 100)
    train_y = np.random.randint(0, n_class, n_data)
    valid_x = np.random.rand(n_data, 100)
    valid_y = np.random.randint(0, n_class, n_data)
    model   = KkGBDT(n_class, mode="xgb", subsample=0.5)
    # public loss
    model.fit(
        train_x, train_y, loss_func="multi:softmax", num_iterations=10,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["mlogloss", "auc"],
        early_stopping_rounds=None, early_stopping_name=0
    )
    # print(model.predict(valid_x, is_softmax=True))
    # custom loss CategoryCE
    model.fit(
        train_x, train_y, loss_func=CategoricalCrossEntropyLoss(n_class), num_iterations=10,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["mlogloss", "auc", CategoricalCrossEntropyLoss(n_class)],
        early_stopping_rounds=None, early_stopping_name=0
    )
    # print(model.predict(valid_x, is_softmax=True))
    # custom loss CE
    train_y = np.random.rand(n_data, n_class)
    valid_y = np.random.rand(n_data, n_class)
    model.fit(
        train_x, train_y, loss_func=CrossEntropyLoss(n_class), num_iterations=10,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=[CrossEntropyLoss(n_class)],
        early_stopping_rounds=None, early_stopping_name=0, 
        stopping_name="ce", stopping_val=3.70, stopping_rounds=1, stopping_is_over=True
    )
    # print(model.predict(valid_x, is_softmax=True))
