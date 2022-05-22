import copy
import numpy as np
import xgboost as xgb
from xgboost.callback import EarlyStopping
from typing import Union, List
from kkgbdt.loss import Loss, LGBCustomObjective, LGBCustomEval
from kkgbdt.dataset import Dataset
from kkgbdt.callbacks import PrintEvalation, TrainStopping
from kkgbdt.util.numpy import softmax
from kkgbdt.util.com import check_type, check_type_list
from kkgbdt.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "KkGBDT",
    "train_xgb",
]


class KkGBDT:
    def __init__(self, num_class: int, mode: str="xgb", is_softmax: bool=False, **kwargs):
        logger.info("START")
        assert isinstance(num_class, int) and num_class > 0
        assert isinstance(mode, str) and mode in ["xgb", "lgb"]
        assert isinstance(is_softmax, bool)
        self.booster    = None
        self.mode       = mode
        self.params = {}
        self.params["num_class"] = num_class
        self.params.update(copy.deepcopy(kwargs))
        self.evals_result = {}
        self.classes_     = np.arange(num_class, dtype=int)
        self.is_softmax   = is_softmax
        logger.info(f"params: {self.params}")
        logger.info("END")
    def fit(
        self, x_train: np.ndarray, y_train: np.ndarray, loss_func: Union[str, Loss]=None, num_boost_round: int=None,
        x_valid: Union[np.ndarray, List[np.ndarray]]=None, y_valid: Union[np.ndarray, List[np.ndarray]]=None,
        loss_func_eval: Union[str, Loss]=None, early_stopping_rounds: int=None, early_stopping_name: Union[int, str]=None,
        stopping_name: str=None, stopping_val: float=None, stopping_rounds: int=None, stopping_is_over: bool=True, stopping_train_time: float=None,
        sample_weight: Union[str, np.ndarray]=None, categorical_features: List[int]=None
    ):
        logger.info("START")
        assert loss_func is not None
        assert num_boost_round is not None
        self.booster = train_xgb(
            copy.deepcopy(self.params), num_boost_round, x_train, y_train, loss_func,
            evals_result=self.evals_result, x_valid=x_valid, y_valid=y_valid, loss_func_eval=loss_func_eval,
            early_stopping_rounds=early_stopping_rounds, early_stopping_name=early_stopping_name,
            stopping_name=stopping_name, stopping_val=stopping_val, stopping_rounds=stopping_rounds, 
            stopping_is_over=stopping_is_over, stopping_train_time=stopping_train_time,
            sample_weight=sample_weight, categorical_features=categorical_features,
        )
        self.set_parameter_after_training()
        logger.info("END")
    def set_parameter_after_training(self):
        self.feature_importances_ = self.booster.get_fscore()
    def predict(self, input: np.ndarray, *args, is_softmax: bool=None, **kwargs):
        logger.info("START")
        output = self.booster.predict(Dataset(input), *args, output_margin=True, **kwargs)
        if is_softmax is None: is_softmax = self.is_softmax
        if is_softmax:
            output = softmax(output)
        logger.info("END")
        return output


def train_xgb(
    # fitting parameter
    params: dict, num_boost_round: int,
    # training data & loss
    x_train: np.ndarray, y_train: np.ndarray, loss_func: Union[str, Loss], evals_result: dict={}, 
    # validation data & loss
    x_valid: Union[np.ndarray, List[np.ndarray]]=None, y_valid: Union[np.ndarray, List[np.ndarray]]=None, loss_func_eval: Union[str, Loss]=None,
    # early stopping parameter
    early_stopping_rounds: int=None, early_stopping_name: Union[int, str]=None,
    stopping_name: str=None, stopping_val: float=None, stopping_rounds: int=None, stopping_is_over: bool=True, stopping_train_time: float=None,
    # option
    sample_weight: Union[str, np.ndarray]=None, categorical_features: List[int]=None
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
    assert isinstance(evals_result, dict)
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
    # dataset option
    if sample_weight is not None:
        assert isinstance(sample_weight, str) or isinstance(sample_weight, np.ndarray)
        if isinstance(sample_weight, str):
            assert sample_weight in ["balanced"]
            assert len(y_train.shape) == 1 and y_train.dtype in [int, np.int16, np.int32, np.int64]
            assert np.unique(y_train).shape == np.bincount(y_train).shape
            sample_weight = np.bincount(y_train)
            sample_weight = sample_weight.min() / sample_weight
            sample_weight = sample_weight[y_train]
        else:
            assert len(sample_weight.shape) == 1 and y_train.shape[0] == sample_weight.shape[0]
    enable_categorical = False
    if categorical_features is not None:
        assert check_type_list(categorical_features, int)
        _categorical_features = np.array(["q"] * x_train.shape[-1]) # "q" is numerical.
        _categorical_features[categorical_features] = "c"
        categorical_features = _categorical_features
        enable_categorical   = True
    # set dataset
    dataset_train = Dataset(x_train, label=y_train, weight=y_train, feature_types=categorical_features, enable_categorical=enable_categorical)
    dataset_valid = [(dataset_train, "train")] + [
        (Dataset(_x_valid, label=_y_valid), f"valid_{i_valid}") for i_valid, (_x_valid, _y_valid) in enumerate(zip(x_valid, y_valid))
    ]
    # loss setting
    _loss_func, _loss_func_eval = None, None
    params["eval_metric"] = []
    if isinstance(loss_func, str):
        params["objective"] = loss_func
    else:
        _loss_func = LGBCustomObjective(loss_func, mode="xgb")
    if loss_func_eval is not None:
        for func_eval in loss_func_eval:
            if isinstance(func_eval, str):
                params["eval_metric"].append(func_eval)
            else:
                if _loss_func_eval is None: _loss_func_eval = []
                _loss_func_eval.append(LGBCustomEval(func_eval, mode="xgb"))
    if _loss_func_eval is not None:
        if len(_loss_func_eval) > 1: logger.warning(f"xgboost's custom metric is supported only one function. so set this metric: {_loss_func_eval[0]}")
        _loss_func_eval = _loss_func_eval[0]
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
    ## train stopping
    if stopping_name is not None:
        assert stopping_val is not None or stopping_train_time is not None
        callbacks.append(
            TrainStopping(stopping_name, stopping_val=stopping_val, stopping_rounds=stopping_rounds, stopping_is_over=stopping_is_over, stopping_train_time=stopping_train_time)
        )
    # train
    model = xgb.train(
        params, dataset_train, 
        num_boost_round=num_boost_round, evals=dataset_valid, obj=_loss_func,
        custom_metric=_loss_func_eval, maximize=False, early_stopping_rounds=None,
        evals_result=evals_result, verbose_eval=False, xgb_model=None,
        callbacks=callbacks
    )
    logger.info("END")
    return model

