import pandas as pd 
from kklogger import set_logger
import numpy as np
from kktestdata import DatasetRegistry
from kkgbdt.model import KkGBDT
from kkgbdt.functions import evaluate_ndcg


LOGGER = set_logger(__name__)


if __name__ == '__main__':
    reg     = DatasetRegistry()
    dataset = reg.create("boatrace_2020_2021")
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataset.load_data(
        format="numpy", split_type="valid", test_size=0.3, valid_size=0.2
    )
    # training
    model = KkGBDT(1, mode="lgb", learning_rate=0.1, max_bin=64, max_depth=-1, eval_at=[3, 6])
    model.fit(
        train_x[:, 1:], train_y, loss_func="lambdarank", num_iterations=200,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval="lambdarank",
        early_stopping_rounds=20, early_stopping_name=1, group_train=train_x[:, 0], group_valid=valid_x[:, 0],
        categorical_features=(
            np.where(np.isin(np.array(dataset.metadata.columns_feature), ["player_no", "number", "exhibition_course"]))[0] - 1
        ).tolist(),
    )
    # evaluation
    ndf_pred = model.predict(test_x[:, 1:])
    benchmark = evaluate_ndcg(ndf_pred,     test_y, is_point_to_rank=True,  k=6, idx_groups=test_x[:, 0])
    pred      = evaluate_ndcg(test_x[:, 1], test_y, is_point_to_rank=False, k=6, idx_groups=test_x[:, 0])
    LOGGER.info(f"pred: {pred.mean()}, base: {benchmark.mean()}", color=["GREEN", "BOLD"])
