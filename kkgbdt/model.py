import copy
import numpy as np
import xgboost as xgb
from xgboost.callback import EarlyStopping
from typing import Union, List
from kkgbdt.loss import Loss, LGBCustomObjective, LGBCustomEval
from kkgbdt.dataset import Dataset
from kkgbdt.callbacks import PrintEvalation
from kkgbdt.util.numpy import softmax
from kkgbdt.util.com import check_type, check_type_list
from kkgbdt.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "KkGBDT",
    "train_xgb",
]


class KkGBDT:
    def __init__(self, num_class: int, params: dict=None, booster=None, is_softmax: bool=False):
        assert isinstance(num_class, int) and num_class > 0
        assert isinstance(is_softmax, bool)
        self.params = {} if params is None else params
        assert isinstance(self.params, dict)
        self.params["num_class"] = num_class
        self.booster    = booster
        self.classes_   = np.arange(num_class, dtype=int)
        self.is_softmax = is_softmax
    def fit(
        self, x_train: np.ndarray, y_train: np.ndarray, loss_func: Union[str, Loss]=None, num_boost_round: int=None,
        x_valid: Union[np.ndarray, List[np.ndarray]]=None, y_valid: Union[np.ndarray, List[np.ndarray]]=None,
        loss_func_eval: Union[str, Loss]=None, early_stopping_rounds: int=None, early_stopping_name: Union[int, str]=None,
        **kwargs
    ):
        assert loss_func is not None
        assert num_boost_round is not None
        self.booster = train_xgb(
            copy.deepcopy(self.params), x_train, y_train, loss_func, num_boost_round,
            x_valid=x_valid, y_valid=y_valid, loss_func_eval=loss_func_eval,
            early_stopping_rounds=early_stopping_rounds, early_stopping_name=early_stopping_name,
            **kwargs
        )
        self.set_parameter_after_training()
    def set_parameter_after_training(self):
        self.feature_importances_ = self.booster.get_fscore()
    def predict(self, input: np.ndarray, *args, is_softmax: bool=None, **kwargs):
        output = self.booster.predict(Dataset(input), *args, output_margin=True, **kwargs)
        if is_softmax is None: is_softmax = self.is_softmax
        if is_softmax:
            output = softmax(output)
        return output


def train_xgb(
    params: dict, x_train: np.ndarray, y_train: np.ndarray, loss_func: Union[str, Loss], num_boost_round: int,
    x_valid: Union[np.ndarray, List[np.ndarray]]=None, y_valid: Union[np.ndarray, List[np.ndarray]]=None,
    loss_func_eval: Union[str, Loss]=None, early_stopping_rounds: int=None, early_stopping_name: Union[int, str]=None,
):
    """
    Params::
        loss_func: custom loss or string
            string:
                see: https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
                multi:softmax, binary:logistic, reg:squarederror, ...
        loss_func_eval:
            custom loss or string.
            string:
                see: https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
                mlogloss, rmse, ...
    """
    logger.info("START")
    assert isinstance(params, dict)
    assert isinstance(x_train, np.ndarray) and len(x_train.shape) == 2
    assert isinstance(y_train, np.ndarray) and len(y_train.shape) in [1, 2]
    assert isinstance(loss_func, str) or isinstance(loss_func, Loss)
    assert isinstance(num_boost_round, int) and num_boost_round > 0
    if x_valid is None: x_valid = []
    if y_valid is None: y_valid = []
    if not isinstance(x_valid, list): x_valid = [x_valid, ]
    if not isinstance(y_valid, list): y_valid = [y_valid, ]
    assert check_type_list(x_valid, np.ndarray)
    assert check_type_list(y_valid, np.ndarray)
    if loss_func_eval is not None:
        if not isinstance(loss_func_eval, list): loss_func_eval = [loss_func_eval, ]
        assert check_type_list(loss_func_eval, [str, Loss])
    # check params
    if params.get("objective")   is not None: logger.raise_error(f"please set 'objective'   parameter to 'loss_func'.")
    if params.get("eval_metric") is not None: logger.raise_error(f"please set 'eval_metric' parameter to 'loss_func_eval'.")
    # set dataset
    dataset_train = Dataset(x_train, label=y_train)
    dataset_valid = [(dataset_train, "train")] + [
        (Dataset(_x_valid, label=_y_valid), f"valid_{i_valid}") for i_valid, (_x_valid, _y_valid) in enumerate(zip(x_valid, y_valid))
    ]
    # loss setting
    _loss_func, _loss_func_eval = None, None
    params["eval_metric"] = []
    if isinstance(loss_func, str):
        params["objective"] = loss_func
    else:
        _loss_func = LGBCustomObjective(loss_func)
    if loss_func_eval is not None:
        for func_eval in loss_func_eval:
            if isinstance(func_eval, str):
                params["eval_metric"].append(func_eval)
            else:
                if _loss_func_eval is None: _loss_func_eval = []
                _loss_func_eval.append(LGBCustomEval(func_eval))
    # callbacks
    callbacks = [PrintEvalation()]
    ## early stopping
    if early_stopping_rounds is not None:
        assert isinstance(early_stopping_rounds, int) and early_stopping_rounds > 0
        assert check_type(early_stopping_name, [int, str])
        if isinstance(early_stopping_name, int):
            assert (early_stopping_name >= 0) and (loss_func_eval is not None) 
            metric_name = loss_func_eval[early_stopping_name]
            if isinstance(metric_name, Loss): metric_name = metric_name.name
        else:
            metric_name = early_stopping_name
        callbacks.append(
            EarlyStopping(early_stopping_rounds, metric_name=metric_name, data_name="valid_0", save_best=True)
        )
    # train
    evals_result = {}
    model = xgb.train(
        params, dataset_train, 
        num_boost_round=num_boost_round, evals=dataset_valid, obj=_loss_func,
        custom_metric=_loss_func_eval, maximize=False, early_stopping_rounds=None,
        evals_result=evals_result, verbose_eval=False, xgb_model=None,
        callbacks=callbacks
    )
    logger.info("END")
    return model

