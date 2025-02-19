import copy
import numpy as np
# local package
from kklogger import set_logger
from .model import KkGBDT
from .loss import Loss


__all__ = [
    "tune_parameter"
]


LOGGER = set_logger(__name__)


def tune_parameter(
    trial, mode: str="xgb", num_class: int=None, n_jobs: int=1, eval_string: str=None,
    x_train: np.ndarray=None, y_train: np.ndarray=None, loss_func: str | Loss=None, num_iterations: int=None,
    x_valid: np.ndarray | list[np.ndarray]=None, y_valid: np.ndarray | list[np.ndarray]=None,
    loss_func_eval: str | Loss=None, early_stopping_rounds: int=None, early_stopping_name: int | str=None,
    train_stopping_val: float=None, train_stopping_rounds: int=None, train_stopping_is_over: bool=True, train_stopping_time: float=None,
    sample_weight: str | np.ndarray=None, categorical_features: list[int]=None,
    params_const = {
        "learning_rate"    : 0.03,
        "num_leaves"       : 100,
        "is_gpu"           : False,
        "random_seed"      : 0,
        "max_depth"        : 0,
        "min_child_samples": 20,
        "subsample"        : 1,
        "colsample_bylevel": 1,
        "colsample_bynode" : 1,
        "max_bin"          : 256, # it's fine. In kkgbdt, reduse -1.
        "min_data_in_bin"  : 5,
    },
    params_search='''{
        "min_child_weight" : trial.suggest_float("min_child_weight", 1e-4, 1e3, log=True),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.001, 0.5),
        "reg_alpha"        : trial.suggest_float("reg_alpha",  1e-4, 1e3, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 1e3, log=True),
        "min_split_gain"   : trial.suggest_float("min_split_gain", 1e-10, 1.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 200),
        "path_smooth"      : trial.suggest_float("path_smooth", 1e-4, 1e2, log=True),
    }'''
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
                num_class=num_class, eval_string='model.evals_result["valid_0"]["mlogloss"][model.booster.best_iteration - 1]',
                x_train=x_train, y_train=y_train, loss_func="multi:softmax", num_iterations=10,
                x_valid=x_valid, y_valid=y_valid, loss_func_eval="mlogloss", 
            )
        >>> study.optimize(func, n_trials=100)
    """
    LOGGER.info("START")
    assert isinstance(eval_string, str)
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert loss_func is not None
    assert isinstance(num_iterations, int)
    params_search = eval(params_search, {}, {"trial": trial})
    params = copy.deepcopy(params_const)
    params.update(params_search)
    model = KkGBDT(num_class, mode=mode, n_jobs=n_jobs, **params)
    model.fit(
        x_train, y_train, loss_func=loss_func, num_iterations=num_iterations, 
        x_valid=x_valid, y_valid=y_valid, loss_func_eval=loss_func_eval,
        early_stopping_rounds=early_stopping_rounds, early_stopping_name=early_stopping_name,
        train_stopping_val=train_stopping_val, train_stopping_rounds=train_stopping_rounds, 
        train_stopping_is_over=train_stopping_is_over, train_stopping_time=train_stopping_time,
        sample_weight=sample_weight, categorical_features=categorical_features,
    )
    LOGGER.info("END")
    return eval(eval_string, {}, {"model": model, "np": np})
