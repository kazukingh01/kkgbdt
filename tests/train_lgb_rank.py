import zipfile
import pandas as pd 
from kklogger import set_logger
import numpy as np
from kkgbdt.model import KkGBDT
from kkgbdt.functions import evaluate_ndcg


LOGGER = set_logger(__name__)


if __name__ == '__main__':
    # data load
    with zipfile.ZipFile('./boatrace_course.zip', 'r') as z:
        with z.open('boatrace_course.csv') as f:
            df = pd.read_csv(f, index_col=False)
    colsoth = ["race_id", "course", "move_real"]
    df      = df.iloc[:, np.where(~(df.columns.isin(colsoth)))[0].tolist() + np.where((df.columns.isin(colsoth)))[0].tolist()]
    df["answer"]  = df["course"].map({1:6, 2:5, 3:4, 4:3, 5:2, 6:1}) # Good course is given good point
    df["race_id"] = df["race_id"].astype(str)
    ndf_bool = (df["race_id"] >= "202001010000").to_numpy(dtype=bool)
    df_valid = df.loc[ndf_bool].copy()
    df_train = df.loc[df.index.isin(df_valid.index) == False].copy()
    train_x  = df_train.iloc[:, :-(len(colsoth) + 1)].to_numpy(dtype=np.float32)
    train_y  = df_train["answer" ].to_numpy(dtype=object)
    train_g  = df_train["race_id"].to_numpy(dtype=object)
    valid_x  = df_valid.iloc[:, :-(len(colsoth) + 1)].to_numpy(dtype=np.float32)
    valid_y  = df_valid["answer" ].to_numpy(dtype=object)
    valid_g  = df_valid["race_id"].to_numpy(dtype=object)
    # training
    model = KkGBDT(1, mode="lgb", learning_rate=0.1, max_bin=64, max_depth=-1, eval_at=[3, 6])
    model.fit(
        train_x, train_y, loss_func="lambdarank", num_iterations=200,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval="lambdarank",
        early_stopping_rounds=20, early_stopping_name=1, group_train=train_g, group_valid=valid_g,
        categorical_features=np.where(df_train.columns.isin(["player_no", "number", "exhibition_course", "jcd"]))[0].tolist(),
    )
    # evaluation
    ndf_pred = model.predict(valid_x)
    df_valid["pred"] = -ndf_pred
    df_valid["pred"] = df_valid.groupby("race_id")["pred"].rank().astype(int)
    pred = df_valid.groupby("race_id")[["exhibition_course", "pred", "course"]].apply(lambda x: evaluate_ndcg(x["pred"             ].to_numpy(), x["course"].to_numpy()))
    base = df_valid.groupby("race_id")[["exhibition_course", "pred", "course"]].apply(lambda x: evaluate_ndcg(x["exhibition_course"].to_numpy(), x["course"].to_numpy()))
    LOGGER.info(f"pred: {pred.mean()}, base: {base.mean()}", color=["GREEN", "BOLD"])
