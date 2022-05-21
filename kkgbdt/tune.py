import copy
from typing import List, Union
import numpy as np
from kkgbdt.model import KkGBDT
from kkgbdt.loss import Loss


__all__ = [
    "parameter_tune"
]


def parameter_tune(
    trial, num_class: int=None, is_gpu: bool=False, eval_string: str=None,
    x_train: np.ndarray=None, y_train: np.ndarray=None, loss_func: Union[str, Loss]=None, num_boost_round: int=None,
    x_valid: Union[np.ndarray, List[np.ndarray]]=None, y_valid: Union[np.ndarray, List[np.ndarray]]=None,
    loss_func_eval: Union[str, Loss]=None, early_stopping_rounds: int=None, early_stopping_name: Union[int, str]=None,
    stopping_name: str=None, stopping_val: float=None, stopping_rounds: int=None, stopping_is_over: bool=True, stopping_train_time: float=None,
):
    """
    Usage::
        >>> 
        eval_string=model.evals_result["train"]["mlogloss"][model.booster.best_iteration]
    """
    assert isinstance(num_class, int)
    assert isinstance(eval_string, str)
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert loss_func is not None
    assert isinstance(num_boost_round, int)
    params_const = {
        "eta"      : 0.03,
        "max_depth": -1,
        "sampling_method": "gradient_based",
        "colsample_bylevel": 1,
        "colsample_bynode": 1,
        "tree_method": "gpu_hist" if is_gpu else "hist",
        "grow_policy": "depthwise",
        "max_leaves": 100,
        "max_bin": 256,
        "single_precision_histogram": True,
        "feature_selector": "cyclic",
        "seed": 0,
    }
    params_search={
        "gamma"            : trial.suggest_uniform('gamma', 0, 1.0),
        "min_child_weight" : trial.suggest_loguniform('min_child_weight', 1e-4, 1e3),
        "subsample"        : trial.suggest_uniform('subsample', 0.5 , 0.99),
        "colsample_bytree" : trial.suggest_loguniform('colsample_bytree', 0.001, 1.0),
        "alpha"            : trial.suggest_loguniform('alpha',  1e-4, 1e3),
        "lambda"           : trial.suggest_loguniform('lambda', 1e-4, 1e3),
    }
    params = copy.deepcopy(params_const)
    params.update(params_search)
    model = KkGBDT(num_class, **params)
    model.fit(
        x_train, y_train, loss_func=loss_func, num_boost_round=num_boost_round, 
        x_valid=x_valid, y_valid=y_valid, loss_func_eval=loss_func_eval,
        early_stopping_rounds=early_stopping_rounds, early_stopping_name=early_stopping_name,
        stopping_name=stopping_name, stopping_val=stopping_val, stopping_rounds=stopping_rounds, 
        stopping_is_over=stopping_is_over, stopping_train_time=stopping_train_time,
    )
    return eval(eval_string, {}, {"model": model, "np": np})
    