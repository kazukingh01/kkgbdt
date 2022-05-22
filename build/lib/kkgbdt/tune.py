import copy
from typing import List, Union
import numpy as np
from kkgbdt.model import KkGBDT
from kkgbdt.loss import Loss


__all__ = [
    "parameter_tune"
]


def parameter_tune(
    trial, num_class: int=None, n_jobs: int=1, is_gpu: bool=False, eval_string: str=None,
    x_train: np.ndarray=None, y_train: np.ndarray=None, loss_func: Union[str, Loss]=None, num_boost_round: int=None,
    x_valid: Union[np.ndarray, List[np.ndarray]]=None, y_valid: Union[np.ndarray, List[np.ndarray]]=None,
    loss_func_eval: Union[str, Loss]=None, early_stopping_rounds: int=None, early_stopping_name: Union[int, str]=None,
    stopping_name: str=None, stopping_val: float=None, stopping_rounds: int=None, stopping_is_over: bool=True, stopping_train_time: float=None,
    sample_weight: Union[str, np.ndarray]=None, categorical_features: List[int]=None,
):
    """
    Usage::
        >>> import optuna
        >>> study = optuna.create_study(study_name="test", storage=f'sqlite:///test.db')
        >>> import numpy as np
        >>> num_class = 5
        >>> x_train = np.random.rand(1000, 10)
        >>> y_train = np.random.randint(0, num_class, 1000)
        >>> x_valid = np.random.rand(1000, 10)
        >>> y_valid = np.random.randint(0, num_class, 1000)
        >>> from functools import partial
        >>> from kkgbdt.tune import parameter_tune
        >>> func = partial(parameter_tune,
                num_class=num_class, eval_string='model.evals_result["valid_0"]["mlogloss"][model.booster.best_iteration]',
                x_train=x_train, y_train=y_train, loss_func="multi:softmax", num_boost_round=10,
                x_valid=x_valid, y_valid=y_valid, loss_func_eval="mlogloss", 
            )
        >>> study.optimize(func, n_trials=100)
    """
    assert isinstance(num_class, int)
    assert isinstance(eval_string, str)
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert loss_func is not None
    assert isinstance(num_boost_round, int)
    params_const = {
        "eta"      : 0.03,
        "max_depth": 0,
        "sampling_method": "uniform",
        "colsample_bylevel": 1,
        "colsample_bynode": 1,
        "tree_method": "gpu_hist" if is_gpu else "hist",
        "grow_policy": "depthwise",
        "max_leaves": 100,
        "max_bin": 256,
        "single_precision_histogram": True,
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
    model = KkGBDT(num_class, n_jobs=n_jobs, **params)
    model.fit(
        x_train, y_train, loss_func=loss_func, num_boost_round=num_boost_round, 
        x_valid=x_valid, y_valid=y_valid, loss_func_eval=loss_func_eval,
        early_stopping_rounds=early_stopping_rounds, early_stopping_name=early_stopping_name,
        stopping_name=stopping_name, stopping_val=stopping_val, stopping_rounds=stopping_rounds, 
        stopping_is_over=stopping_is_over, stopping_train_time=stopping_train_time,
        sample_weight=sample_weight, categorical_features=categorical_features,
    )
    return eval(eval_string, {}, {"model": model, "np": np})
    