import numpy as np
from kkgbdt.model import KkGBDT


if __name__ == "__main__":
    n_class, n_data = 5, 100
    train_x = np.random.rand(n_data, 10)
    train_y = np.random.randint(0, n_class, n_data)
    valid_x = np.random.rand(n_data, 10)
    valid_y = np.random.randint(0, n_class, n_data)
    model   = KkGBDT(n_class)
    model.fit(
        train_x, train_y, loss_func="multi:softmax", num_boost_round=10,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=["mlogloss", "auc"],
        early_stopping_rounds=3, early_stopping_name=0
    )
    model.predict(valid_x, is_softmax=True)