import numpy as np
import pandas as pd
from kkgbdt.model import KkGBDT
from kkgbdt.trace import KkTracer
    
def func_rmse_value(y, penalty_l2: float=0):
    return -1 * (-2 * y.sum()) / (2 * y.shape[0] + penalty_l2)
def func_rmse_split_gain(y, penalty_l1: float=0, penalty_l2: float=0):
    return ((-2 * y.sum()) ** 2) / (2 * y.shape[0] + penalty_l2)

if __name__ == "__main__":
    n_class, n_data = 1, 10000
    train_x = np.random.rand(n_data, 100)
    train_y = np.random.rand(n_data)
    test_x  = train_x
    test_y  = train_y
    model   = KkGBDT(n_class, mode="lgb", subsample=1.0, colsample_bytree=0.5, max_depth=10, max_bin=4, reg_lambda=0.0)
    model.fit(train_x, train_y, loss_func="mse", num_iterations=100, loss_func_eval=["mse"])
    booster = model.booster
    booster.__class__ = KkTracer
    booster.create_tree_dataframe()
    print(f"tree structure: \n{booster.tracer_dftree.iloc[0]}")
    print(f"calc: {func_rmse_value(train_y, penalty_l2=booster.params['lambda_l2'])}")
    leaf_df_index = booster.predict_leaf_index(test_x)
    leaf_v, ndf_path, ndf_thre, ndf_gain, ndf_col = booster.interpret_path(leaf_df_index)
    for i_index in np.unique(ndf_path[:, 1, :1])[0:1]:
        ndf_bool = (ndf_path[:, 1, :1] == i_index)
        left  = func_rmse_split_gain(test_y[ ndf_bool.reshape(-1)])
        right = func_rmse_split_gain(test_y[~ndf_bool.reshape(-1)])
        root  = func_rmse_split_gain(test_y)
        print(f"root: {root}, left or right: {left}, {right}")
        print(f"split value: {(left + right - root) / 2}")
    print(booster.create_path_matrix(booster.predict_leaf_index(test_x[:1])))
