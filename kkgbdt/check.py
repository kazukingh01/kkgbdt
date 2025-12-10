import copy
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from kklogger import set_logger
from .loss import Loss
from .com import check_type_list


LOGGER = set_logger(__name__)


MODE = ["xgb", "lgb", "cat"]
MST_OBJECTIVE = {
    "binary": {"xgb": "binary:logistic",      "lgb": "binary",     "cat": "Logloss"},
    "multi":  {"xgb": "multi:softmax",        "lgb": "multiclass", "cat": "MultiClass"},
    "reg":    {"xgb": "reg:squarederror",     "lgb": "regression", "cat": "RMSE"},
    "huber":  {"xgb": "reg:pseudohubererror", "lgb": "huber",      "cat": "Huber"},
    "rank":   {"xgb": "rank:ndcg",            "lgb": "lambdarank", "cat": "YetiRank"},
}
MST_METRIC = {
    "binary": {"xgb": "logloss",  "lgb": "binary_logloss", "cat": "Logloss"},
    "multi":  {"xgb": "mlogloss", "lgb": "multi_logloss",  "cat": "MultiClass"},
    "reg":    {"xgb": "rmse",     "lgb": "rmse",           "cat": "RMSE"},
    "huber":  {"xgb": "mphe",     "lgb": "huber",          "cat": "Huber"},
    "rank":   {"xgb": "ndcg",     "lgb": "ndcg",           "cat": "NDCG"},
    "acc":    {"xgb": None,       "lgb": None,             "cat": "Accuracy"},
    "auc":    {"xgb": "auc",      "lgb": "auc",            "cat": "AUC"},
}


def check_mode(mode: str):
    assert isinstance(mode, str) and mode in MODE

def check_other(params: dict, num_iterations: int, evals_result: dict):
    assert isinstance(params, dict)
    assert isinstance(num_iterations, int) and num_iterations > 0
    assert isinstance(evals_result, dict)

def check_inputs(
    x_train: np.ndarray, y_train: np.ndarray,
    x_valid: np.ndarray | list[np.ndarray] | None=None, y_valid: np.ndarray | list[np.ndarray] | None=None
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    assert isinstance(x_train, np.ndarray) and len(x_train.shape) == 2
    assert isinstance(y_train, np.ndarray) and len(y_train.shape) in [1, 2]
    if x_valid is None: x_valid = []
    if y_valid is None: y_valid = []
    if not isinstance(x_valid, list): x_valid = [x_valid, ]
    if not isinstance(y_valid, list): y_valid = [y_valid, ]
    assert check_type_list(x_valid, np.ndarray)
    assert check_type_list(y_valid, np.ndarray)
    for _x_valid, _y_valid in zip(x_valid, y_valid):
        assert len(x_train.shape) == len(_x_valid.shape)
        assert len(y_train.shape) == len(_y_valid.shape)
        if len(x_train.shape) > 1:
            assert x_train.shape[-1]  == _x_valid.shape[-1]
        if len(y_train.shape) > 1:
            assert y_train.shape[-1]  == _y_valid.shape[-1]
    return x_train, y_train, x_valid, y_valid

def str_loss_to_metric(loss_func: str, mode: str) -> str:
    assert isinstance(loss_func, str)
    check_mode(mode)
    _type = None
    for x, y in MST_OBJECTIVE.items():
        if y[mode] == loss_func:
            _type = x
    assert _type is not None, "loss_func is not found in MST_OBJECTIVE"
    return MST_METRIC[_type][mode]

def check_loss_string_catboost(loss_func: str, is_metric: bool=False) -> str:
    """
    loss_func: huber(delta=1.0,use_weights=false)
    """
    assert isinstance(loss_func, str)
    if "(" in loss_func and ")" in loss_func:
        _loss   = loss_func.split("(")[0]
        _params = loss_func.split("(")[1].split(")")[0]
        _params = ":".join([x.strip() for x in _params.split(",")])
        if is_metric:
            assert _loss in MST_METRIC
            return f"{MST_METRIC[_loss]['cat']}:{_params}"
        else:
            assert _loss in MST_OBJECTIVE
            return f"{MST_OBJECTIVE[_loss]['cat']}:{_params}"
    else:
        if is_metric:
            assert loss_func in MST_METRIC
            return MST_METRIC[loss_func]['cat']
        else:
            assert loss_func in MST_OBJECTIVE
            return MST_OBJECTIVE[loss_func]['cat']

def check_loss_func(
    loss_func: str | Loss, mode: str, loss_func_eval: str | list[str | Loss] | None = None, x_valid: list[np.ndarray] = None
) -> tuple[str | Loss, list[str | Loss]]:
    """
    loss_func_eval is allowed "__copy__" string, which means to copy the loss function.
    """
    assert isinstance(loss_func, (str, Loss))
    check_mode(mode)
    assert x_valid is not None and isinstance(x_valid, list)
    assert check_type_list(x_valid, np.ndarray)
    _loss_func: str | Loss = None
    _loss_func_eval: list[str | Loss] = None
    if isinstance(loss_func, str):
        if mode == "cat":
            _loss_func = check_loss_string_catboost(loss_func, is_metric=False)
        else:
            assert loss_func in MST_OBJECTIVE
            _loss_func = MST_OBJECTIVE[loss_func][mode]
    else:
        _loss_func = loss_func
    if loss_func_eval is not None:
        _loss_func_eval = []
        assert len(x_valid) > 0
        if not isinstance(loss_func_eval, list): loss_func_eval = [loss_func_eval, ]
        assert check_type_list(loss_func_eval, [str, Loss])
        for _loss in loss_func_eval:
            if isinstance(loss_func, str) and mode == "xgb" and isinstance(_loss, Loss):
                LOGGER.warning(
                    "When loss function is normal and eval function is custom, input of convert_xgb_input(cls, y_pred: np.ndarray, data: xgb.DMatrix) " +
                    "is strange. The input is like y_pred [4. 4. 1. ... 0. 0. 0.] even classification, it's not probability. So in this case, " + 
                    "custom function for evalation is ignored."
                )
                continue
            if isinstance(_loss, str):
                if _loss == "__copy__":
                    assert isinstance(loss_func, Loss)
                    _loss_func_eval.append(copy.deepcopy(loss_func))
                else:
                    if mode == "cat":
                        _loss_func_eval.append(check_loss_string_catboost(_loss, is_metric=True))
                    else:
                        assert _loss in MST_METRIC
                        _loss_func_eval.append(MST_METRIC[_loss][mode])
            else:
                _loss_func_eval.append(copy.deepcopy(_loss))
    else:
        if len(x_valid) > 0:
            if isinstance(loss_func, Loss):
                _loss_func_eval = [copy.deepcopy(loss_func), ]
            else:
                if mode == "cat":
                    _loss_func_eval = [check_loss_string_catboost(loss_func, is_metric=True), ]
                else:
                    _loss_func_eval = [MST_METRIC[loss_func][mode], ]
    return _loss_func, _loss_func_eval

def check_early_stopping(
    early_stopping_rounds: int, early_stopping_idx: int, mode: str, loss_func_eval: list[str | Loss]
) -> int | str:
    assert isinstance(early_stopping_rounds, int) and early_stopping_rounds > 0
    assert isinstance(early_stopping_idx, int)
    check_mode(mode)
    assert isinstance(loss_func_eval, list) and len(loss_func_eval) > 0
    assert early_stopping_idx >= 0
    if mode == "lgb":
        return early_stopping_idx # lgb は int で処理する. loss_func_eval は気にしない
    elif mode == "xgb":
        assert early_stopping_idx < len(loss_func_eval)
        metric_name = loss_func_eval[early_stopping_idx]
        if isinstance(metric_name, Loss):
            metric_name = metric_name.name
        return metric_name
    else:
        pass
    return early_stopping_idx

def check_train_stopping(
    train_stopping_val: float | int | None, train_stopping_time: float | int | None,
    train_stopping_rounds: int, train_stopping_is_over: bool
):
    assert train_stopping_val is not None or train_stopping_time is not None
    assert train_stopping_val  is None or isinstance(train_stopping_val,  (int, float))
    assert train_stopping_time is None or isinstance(train_stopping_time, (int, float))
    assert isinstance(train_stopping_rounds, int) and train_stopping_rounds > 0
    assert isinstance(train_stopping_is_over, bool)

def check_and_compute_sample_weight(
    sample_weight: str | np.ndarray | list[str | np.ndarray], y_target: np.ndarray
) -> np.ndarray:
    """
    sample_weight が複数ある場合は、step by step で計算する
    例えば、["balanced", (np.arange(100) / 100) ** (1/2)] のようになっている場合は
    class で balance させつつ、前半のデータは重みを相対的に薄くしている
    """
    assert sample_weight is not None
    assert isinstance(sample_weight, (str, np.ndarray)) or isinstance(sample_weight, list)
    if not isinstance(sample_weight, list): sample_weight = [sample_weight, ]
    assert check_type_list(sample_weight, [str, np.ndarray])
    assert isinstance(y_target, np.ndarray)
    _sample_weight = np.ones(y_target.shape[0]).astype(float)
    LOGGER.info(f"sample weight before: {sample_weight}")
    for sw in sample_weight:
        if isinstance(sw, str):
            assert sw in ["balanced"]
            assert len(y_target.shape) == 1 and y_target.dtype in [int, np.int16, np.int32, np.int64]
            assert np.unique(y_target).shape == np.bincount(y_target).shape
            _sample_weight = _sample_weight * compute_sample_weight("balanced", y_target) # https://github.com/microsoft/LightGBM/blob/a5285985992ebd376c356a0ac1d10a190338550b/python-package/lightgbm/sklearn.py#L771
        else:
            sw: np.ndarray
            assert len(sw.shape) == 1 and y_target.shape[0] == sw.shape[0]
            assert sw.dtype in [float, np.float16, np.float32, np.float64]
            _sample_weight = _sample_weight * sw
    LOGGER.info(f"sample weight after:  {_sample_weight}")
    return _sample_weight

def check_groups_for_rank(
    group_train: np.ndarray | None=None, group_valid: np.ndarray | list[np.ndarray] | None=None,
    x_valid: list[np.ndarray] = None
) -> tuple[np.ndarray | None, list[np.ndarray | None]]:
    """
    group_train is list for group id 
    [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
    """
    if group_train is not None:
        assert isinstance(group_train, np.ndarray)
        assert len(group_train.shape) == 1
    assert isinstance(x_valid, list)
    if len(x_valid) > 0:
        assert check_type_list(x_valid, np.ndarray)
    if group_valid is not None:
        assert isinstance(group_valid, (np.ndarray, list))
        if not isinstance(group_valid, list): group_valid = [group_valid, ]
        assert check_type_list(group_valid, np.ndarray)
        assert all(len(g.shape) == 1 for g in group_valid)
        if len(x_valid) > 0:
            assert len(x_valid) == len(group_valid)
            assert all(x.shape[0] == y.shape[0] for x, y in zip(x_valid, group_valid))
    else:
        group_valid = [None, ] * len(x_valid)
    return group_train, group_valid