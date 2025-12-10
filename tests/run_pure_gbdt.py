import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from kktestdata import DatasetRegistry


def eval_binary(y_true, y_pred_proba):
    """Accuracy for binary classification."""
    y_pred = (y_pred_proba >= 0.5).astype(int)
    acc = (y_true == y_pred).mean()
    return acc

def eval_multiclass(y_true, y_pred_proba):
    """Accuracy for multi-class classification."""
    y_pred = np.argmax(y_pred_proba, axis=1)
    acc = (y_true == y_pred).mean()
    return acc

def eval_regression(y_true, y_pred):
    """RMSE for regression."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return rmse

def ndcg_at_k(y_true, y_score, group, k=5):
    """Simple NDCG@k for ranking, averaged over groups."""
    start = 0
    ndcgs = []
    for gsize in group:
        end = start + gsize
        rel = y_true[start:end]
        score = y_score[start:end]
        order = np.argsort(-score)
        rel_sorted = rel[order][:k]
        gains = (2.0 ** rel_sorted - 1.0)
        discounts = np.log2(np.arange(2, len(rel_sorted) + 2))
        dcg = float(np.sum(gains / discounts))
        ideal_rel = np.sort(rel)[::-1][:k]
        ideal_gains = (2.0 ** ideal_rel - 1.0)
        ideal_dcg = float(np.sum(ideal_gains / discounts))
        ndcgs.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
        start = end
    return float(np.mean(ndcgs))



if __name__ == "__main__":
    reg = DatasetRegistry()
    PARAMS_LGB = {"learning_rate": 0.1}
    PARAMS_XGB = {"eta": 0.1}
    PARAMS_CB  = {"learning_rate": 0.1, "iterations": 100, "logging_level": "Verbose"}
    CALLBACKS_LGB = [
        lgb.early_stopping(stopping_rounds=20, verbose=True),
        lgb.log_evaluation(1),
    ]


    # binary
    dataset = reg.create("Bank_Customer_Churn")
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataset.load_data(
        format="numpy", split_type="valid", test_size=0.3, valid_size=0.2
    )
    print(dataset.to_display())
    print(dataset.metadata.label_mapping_feature)
    ## LightGBM
    ds_lgb_train = lgb.Dataset(train_x, label=train_y)
    ds_lgb_valid = lgb.Dataset(valid_x, label=valid_y, reference=ds_lgb_train)
    ins_lgb  = lgb.train(
        PARAMS_LGB | {"objective": "binary"}, 
        ds_lgb_train, num_boost_round=100, valid_sets=[ds_lgb_valid], callbacks=CALLBACKS_LGB,
    )
    pred_lgb = ins_lgb.predict(test_x)
    print("LightGBM binary acc:", eval_binary(test_y, pred_lgb))
    ## XGBoost
    ds_xgb_train = xgb.DMatrix(train_x, label=train_y)
    ds_xgb_valid = xgb.DMatrix(valid_x, label=valid_y)
    ins_xgb = xgb.train(
        PARAMS_XGB | {"objective": "binary:logistic"}, ds_xgb_train, num_boost_round=100, 
        evals=[(ds_xgb_valid, "valid")], early_stopping_rounds=20, verbose_eval=True
    )
    pred_xgb = ins_xgb.predict(xgb.DMatrix(test_x), output_margin=False)
    print("XGBoost binary acc:", eval_binary(test_y, pred_xgb))
    ## CatBoost
    ds_cb_train = cb.Pool(train_x, train_y)
    ds_cb_valid = cb.Pool(valid_x, valid_y)
    ins_cb = cb.train(
        pool=ds_cb_train, eval_set=ds_cb_valid, early_stopping_rounds=20,
        params=PARAMS_CB | {"loss_function": "Logloss", },
    )
    pred_cb = ins_cb.predict(cb.Pool(test_x), prediction_type="Probability")
    print("CatBoost binary acc:", eval_binary(test_y, pred_cb[:, 1]))


    # multi-class
    dataset = reg.create("gas-drift")
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataset.load_data(
        format="numpy", split_type="valid", test_size=0.3, valid_size=0.2
    )
    print(dataset.to_display())
    print(dataset.metadata.label_mapping_feature)
    num_class = int(np.unique(train_y).shape[0])
    ## LightGBM
    ds_lgb_train = lgb.Dataset(train_x, label=train_y)
    ds_lgb_valid = lgb.Dataset(valid_x, label=valid_y, reference=ds_lgb_train)
    ins_lgb  = lgb.train(
        PARAMS_LGB | {"objective": "multiclass", "num_class": num_class, "metric": "multi_logloss"},
        ds_lgb_train, num_boost_round=100, valid_sets=[ds_lgb_valid], callbacks=CALLBACKS_LGB,
    )
    pred_lgb = ins_lgb.predict(test_x)
    print("LightGBM multi-class acc:", eval_multiclass(test_y, pred_lgb))
    ## XGBoost
    ds_xgb_train = xgb.DMatrix(train_x, label=train_y)
    ds_xgb_valid = xgb.DMatrix(valid_x, label=valid_y)
    ins_xgb = xgb.train(
        PARAMS_XGB | {"objective": "multi:softmax", "num_class": num_class, "eval_metric": "mlogloss"},
        ds_xgb_train, num_boost_round=100, evals=[(ds_xgb_valid, "valid")],
        early_stopping_rounds=20, verbose_eval=True
    )
    pred_xgb = ins_xgb.predict(xgb.DMatrix(test_x), output_margin=True)
    print("XGBoost multi-class acc:", eval_multiclass(test_y, pred_xgb))
    ## CatBoost
    ds_cb_train = cb.Pool(train_x, train_y)
    ds_cb_valid = cb.Pool(valid_x, valid_y)
    ins_cb = cb.train(
        pool=ds_cb_train, eval_set=ds_cb_valid, early_stopping_rounds=20,
        params=PARAMS_CB | {"loss_function": "MultiClass", },
    )
    pred_cb = ins_cb.predict(cb.Pool(test_x), prediction_type="Probability")
    print("CatBoost multi-class acc:", eval_multiclass(test_y, pred_cb))


    # Regression
    train_x, train_y, valid_x, valid_y, test_x, test_y = reg.create("diamonds").load_data(
        format="numpy", split_type="valid", test_size=0.3, valid_size=0.2
    )
    print(dataset.to_display())
    print(dataset.metadata.label_mapping_feature)
    ## LightGBM
    ds_lgb_train = lgb.Dataset(train_x, label=train_y)
    ds_lgb_valid = lgb.Dataset(valid_x, label=valid_y, reference=ds_lgb_train)
    ins_lgb  = lgb.train(
        PARAMS_LGB | {"objective": "regression", "metric": "rmse"},
        ds_lgb_train, num_boost_round=100, valid_sets=[ds_lgb_valid], callbacks=CALLBACKS_LGB,
    )
    pred_lgb = ins_lgb.predict(test_x)
    print("LightGBM RMSE:", eval_regression(test_y, pred_lgb))
    ## XGBoost
    ds_xgb_train = xgb.DMatrix(train_x, label=train_y)
    ds_xgb_valid = xgb.DMatrix(valid_x, label=valid_y)
    ins_xgb = xgb.train(
        PARAMS_XGB | {"objective": "reg:squarederror", "eval_metric": "rmse"},
        ds_xgb_train, num_boost_round=100, evals=[(ds_xgb_valid, "valid")],
        early_stopping_rounds=20, verbose_eval=True
    )
    pred_xgb = ins_xgb.predict(xgb.DMatrix(test_x), output_margin=False)
    print("XGBoost RMSE:", eval_regression(test_y, pred_xgb))
    ## CatBoost
    ds_cb_train = cb.Pool(train_x, train_y)
    ds_cb_valid = cb.Pool(valid_x, valid_y)
    ins_cb = cb.train(
        pool=ds_cb_train, eval_set=ds_cb_valid, early_stopping_rounds=20,
        params=PARAMS_CB | {"loss_function": "RMSE", },
    )
    pred_cb = ins_cb.predict(cb.Pool(test_x), prediction_type="RawFormulaVal")
    print("CatBoost RMSE:", eval_regression(test_y, pred_cb))


    # Rank
    dataset = reg.create("boatrace_2020_2021")
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataset.load_data(
        format="numpy", split_type="valid", test_size=0.3, valid_size=0.2
    )
    print(dataset.to_display())
    print(dataset.metadata.label_mapping_feature)
    dict_train = {x: i for i, x in enumerate(np.sort(np.unique(train_x[:, 0])))}
    dict_valid = {x: i for i, x in enumerate(np.sort(np.unique(valid_x[:, 0])))}
    dict_test  = {x: i for i, x in enumerate(np.sort(np.unique(test_x[ :, 0])))}
    group_train = np.vectorize(lambda x: dict_train[x])(train_x[:, 0])
    group_valid = np.vectorize(lambda x: dict_valid[x])(valid_x[:, 0])
    group_test  = np.vectorize(lambda x: dict_test[ x])(test_x[ :, 0])
    assert train_x.shape[0] % 6 == 0
    assert valid_x.shape[0] % 6 == 0
    assert test_x. shape[0] % 6 == 0
    ## LightGBM
    ds_lgb_train = lgb.Dataset(train_x[:, 1:], label=train_y, group=([6] * (train_x.shape[0] // 6)))
    ds_lgb_valid = lgb.Dataset(valid_x[:, 1:], label=valid_y, group=([6] * (valid_x.shape[0] // 6)), reference=ds_lgb_train)
    ins_lgb  = lgb.train(
        PARAMS_LGB | {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [4]},
        ds_lgb_train, num_boost_round=100, valid_sets=[ds_lgb_valid], callbacks=CALLBACKS_LGB,
    )
    pred_lgb = ins_lgb.predict(test_x[:, 1:])
    print("LightGBM ranking NDCG@4:", ndcg_at_k(test_y, pred_lgb, ([6] * (test_x.shape[0] // 6)), k=4))
    ## XGBoost
    ds_xgb_train = xgb.DMatrix(train_x, label=train_y)
    ds_xgb_valid = xgb.DMatrix(valid_x, label=valid_y)
    ds_xgb_train.set_group(([6] * (train_x.shape[0] // 6)))
    ds_xgb_valid.set_group(([6] * (valid_x.shape[0] // 6)))
    ins_xgb = xgb.train(
        PARAMS_XGB | {"objective": "rank:ndcg", "eval_metric": "ndcg"},
        ds_xgb_train, num_boost_round=100, evals=[(ds_xgb_valid, "valid")],
        early_stopping_rounds=20, verbose_eval=True
    )
    pred_xgb = ins_xgb.predict(xgb.DMatrix(test_x), output_margin=False)
    print("XGBoost ranking NDCG@4:", ndcg_at_k(test_y, pred_xgb, ([6] * (test_x.shape[0] // 6)), k=4))
    ## CatBoost
    ds_cb_train = cb.Pool(train_x, train_y, group_id=group_train)
    ds_cb_valid = cb.Pool(valid_x, valid_y, group_id=group_valid)
    ins_cb = cb.train(
        pool=ds_cb_train, eval_set=ds_cb_valid, early_stopping_rounds=20,
        params=PARAMS_CB | {"loss_function": "YetiRank", },
    )
    pred_cb = ins_cb.predict(cb.Pool(test_x, group_id=group_test))
    print("CatBoost ranking NDCG@4:", ndcg_at_k(test_y, pred_cb, ([6] * (test_x.shape[0] // 6)), k=4))
