import os, copy, time, json, base64, json
import numpy as np
from typing import Any
from functools import partial
from dataclasses import dataclass
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from xgboost.callback import EarlyStopping
from lightgbm.callback import record_evaluation
# local package
from .check import (
    check_inputs, check_loss_func, check_early_stopping, check_train_stopping, check_other, 
    check_and_compute_sample_weight, check_groups_for_rank, check_mode, str_loss_to_metric,
    is_prob_from_loss_func
)
from .loss import Loss, LGBCustomObjective, LGBCustomEval
from .dataset import DatasetLGB, DatasetCB, create_xgb_dataset_with_group
from .callbacks import (
    PrintEvalation, TrainStopping, log_evaluation, callback_stop_training, callback_best_iter, create_callbacks_cb
)
from .functions import softmax, sigmoid
from .com import check_type, check_type_list
from kklogger import set_logger, set_loglevel, LoggingNameException


LOGGER = set_logger(__name__)


__all__ = [
    "KkGBDT",
    "train_xgb",
    "train_lgb",
    "train_cat",
    "alias_parameter",
    "set_all_loglevel",
]


@dataclass
class ParamsTraining:
    # fitting parameter
    params: dict
    num_iterations: int
    # training data & loss
    x_train: np.ndarray | str
    y_train: np.ndarray | None
    loss_func: str | Loss
    # validation data & loss
    x_valid: np.ndarray | list[np.ndarray] | str | list[str] | None=None
    y_valid: np.ndarray | list[np.ndarray] | None=None
    loss_func_eval: str | Loss | list[str | Loss]=None
    # early stopping parameter
    early_stopping_rounds: int=None
    early_stopping_idx: int=None
    train_stopping_val: float=None
    train_stopping_rounds: int=None
    train_stopping_is_over: bool=True
    train_stopping_time: float=None
    # option
    sample_weight: None | np.ndarray=None
    categorical_features: list[int]=None
    group_train: None | np.ndarray=None
    group_valid: None | np.ndarray | list[np.ndarray]=None
    encode_type: int | None=None

class KkGBDT:
    def __init__(
        self, num_class: int, mode: str="xgb", is_softmax: bool=None, save_dataset: str=None,
        learning_rate: float=0.1, num_leaves: int | None=None, n_jobs: int=-1, is_gpu: bool=False, 
        random_seed: int=0, max_depth: int | None=6, min_child_samples: int | None=None, min_child_weight: float | None=None,
        subsample: float | None=None, colsample_bytree: float | None=None, colsample_bylevel: float | None=None, colsample_bynode: float | None=None,
        reg_alpha: float | None=None, reg_lambda: float | None=None, min_split_gain: float | None=None, max_bin: int=256, 
        min_data_in_bin: int=5, path_smooth: float | None=None, multi_strategy: str | None=None, 
        grow_policy: str | None=None, is_extratree: bool | None=None, verbosity: int | None=None, **kwargs
    ):
        LOGGER.info("START")
        check_mode(mode)
        assert is_softmax   is None or isinstance(is_softmax, bool)
        assert save_dataset is None or isinstance(save_dataset, str)
        self.booster = None
        self.mode    = mode
        self.params  = alias_parameters(
            mode=self.mode, num_class=num_class, learning_rate=learning_rate, num_leaves=num_leaves, n_jobs=n_jobs, is_gpu=is_gpu, 
            random_seed=random_seed, max_depth=max_depth, min_child_samples=min_child_samples, min_child_weight=min_child_weight,
            subsample=subsample, colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode, reg_alpha=reg_alpha, reg_lambda=reg_lambda, min_split_gain=min_split_gain,
            max_bin=max_bin, min_data_in_bin=min_data_in_bin, path_smooth=path_smooth, multi_strategy=multi_strategy, 
            grow_policy=grow_policy, is_extratree=is_extratree, verbosity=verbosity
        )
        self.params.update(copy.deepcopy(kwargs))
        self.params_pkg          = {}
        self.evals_result        = {}
        self.classes_            = [int(x) for x in np.arange(num_class, dtype=int)]
        self.is_softmax          = is_softmax
        self.time_train          = 0
        self.best_iteration      = -1
        self.total_iteration     = -1
        self.feature_importances = []
        self.loss                = None
        self.inference           = None
        self.booster             = None
        self.save_dataset        = save_dataset
        if mode == "xgb":
            self.train_func   = _train_xgb
            self.predict_func = self.predict_xgb
        elif mode == "lgb":
            self.train_func   = partial(_train_lgb, save_dataset=self.save_dataset)
            self.predict_func = self.predict_lgb
        else:
            self.train_func   = _train_cat
            self.predict_func = self.predict_cat
        LOGGER.info(f"params: {self.params}")
        LOGGER.info("END")
    def __str__ (self):
        return f"{self.__class__.__name__}({self.mode}, iters={self.total_iteration}, best={self.best_iteration}, loss={self.loss})"
    def __repr__(self):
        return self.__str__()
    def copy(self):
        instance = copy.deepcopy(self)
        for x in ["best_iteration", "best_score"]:
            if hasattr(self.booster, x):
                setattr(instance.booster, x, copy.deepcopy(getattr(self.booster, x)))
        return instance
    def fit(
        self, x_train: np.ndarray | str, y_train: np.ndarray | None, loss_func: str | Loss=None, num_iterations: int=None,
        x_valid: np.ndarray | list[np.ndarray] | str | list[str]=None, y_valid: np.ndarray | list[np.ndarray] | None=None,
        loss_func_eval: str | Loss | list[str | Loss]=None, early_stopping_rounds: int=None, early_stopping_idx: int | str=None,
        train_stopping_val: float=None, train_stopping_rounds: int=None, train_stopping_is_over: bool=True, train_stopping_time: float=None,
        sample_weight: str | np.ndarray | list[str | np.ndarray]=None, categorical_features: list[int]=None,
        group_train: None | np.ndarray=None, group_valid: None | np.ndarray | list[np.ndarray]=None, random_seed: int=None,
    ):
        LOGGER.info("START", color=["BOLD", "GREEN"])
        # check
        check_other(self.params, num_iterations, self.evals_result)
        # check inputs & convert
        x_train, y_train, x_valid, y_valid = check_inputs(x_train, y_train, x_valid, y_valid)
        if isinstance(x_train, str):
            assert self.mode == "lgb"
        # check loss function & convert
        loss_func, loss_func_eval, is_prob, encode_type = check_loss_func(loss_func, self.mode, loss_func_eval, x_valid)
        self.loss = loss_func
        if self.is_softmax is None:
            LOGGER.info(f"set is_softmax: {is_prob}")
            self.is_softmax = is_prob
        if isinstance(self.loss, Loss) and hasattr(self.loss, "inference"):
            self.inference = self.loss.inference
        # check sample weight & convert
        if sample_weight is not None and y_train is not None:
            sample_weight = check_and_compute_sample_weight(sample_weight, y_train)
        # check early stopping
        if len(x_valid) == 0:
            assert early_stopping_idx    is None, "Don't set early_stopping_idx when there is no validation data."
            assert early_stopping_rounds is None, "Don't set early_stopping_rounds when there is no validation data."            
        # check groups for rank
        group_train, group_valid = check_groups_for_rank(group_train, group_valid=group_valid, x_valid=x_valid)
        # training
        time_start = time.time()
        self.booster = self.train_func(
            ParamsTraining(
                params=(copy.deepcopy(self.params) | (set_random_seed(self.mode, random_seed) if random_seed is not None else {})),
                num_iterations=num_iterations,
                x_train=x_train,
                y_train=y_train,
                loss_func=self.loss,
                x_valid=x_valid,
                y_valid=y_valid,
                loss_func_eval=loss_func_eval,
                early_stopping_rounds=early_stopping_rounds,
                early_stopping_idx=early_stopping_idx,
                train_stopping_val=train_stopping_val,
                train_stopping_rounds=train_stopping_rounds,
                train_stopping_is_over=train_stopping_is_over,
                train_stopping_time=train_stopping_time,
                sample_weight=sample_weight,
                categorical_features=categorical_features,
                group_train=group_train,
                group_valid=group_valid,
                encode_type=encode_type,
            ),
            evals_result=self.evals_result,
        )
        self.time_train = time.time() - time_start
        # save params
        if self.mode == "xgb":
            self.params_pkg = json.loads(self.booster.save_config())
        elif self.mode == "lgb":
            self.params_pkg = copy.deepcopy(self.booster.params)
        else:
            self.params_pkg = copy.deepcopy(self.booster.get_all_params())
        # total iteration
        dictwk = self.evals_result.get("train", {})
        if dictwk:
            self.total_iteration = len(dictwk.get(list(dictwk.keys())[0]))
        else:
            self.total_iteration = num_iterations
        if self.mode == "xgb":
            self.total_iteration += -1
        # best iteration
        if self.mode == "xgb":
            try:
                self.best_iteration = self.booster.best_iteration
                if self.best_iteration is not None and self.best_iteration > 0:
                    self.best_iteration += 1
            except AttributeError:
                self.best_iteration = 0
        elif self.mode == "lgb":
            self.best_iteration = self.booster.best_iteration # This is counted from 0
        else:
            self.best_iteration = self.booster.get_best_iteration()
        LOGGER.info(f"best iteration is {self.best_iteration}. # 0 means maximum iteration is selected.")
        # additional prosessing for custom loss
        if isinstance(self.loss, Loss) and hasattr(self.loss, "extra_processing"):
            output = self.predict_func(x_train)
            self.loss.extra_processing(output, y_train)
        self.set_parameter_after_training()
        LOGGER.info(f"Time: {self.time_train} s.")
        LOGGER.info("END", color=["BOLD", "GREEN"])
    def set_parameter_after_training(self):
        if self.mode == "xgb":
            dictwk = {f"f{i}": 0 for i in range(self.booster.num_features())}
            dictwk = dictwk | self.booster.get_fscore()
            self.feature_importances = list(dictwk.values())
        elif self.mode == "lgb":
            self.feature_importances = self.booster.feature_importance().tolist()
        else:
            try:
                self.feature_importances = self.booster.get_feature_importance().tolist()
            except cat.CatBoostError as e:
                LOGGER.warning(f"Error at get_feature_importance(): {e}")
                self.feature_importances = []
    def predict(self, input: np.ndarray, *args, is_softmax: bool | None=None, iteration_at: int | None=None, **kwargs):
        LOGGER.info(f"args: {args}, is_softmax: {is_softmax}, kwargs: {kwargs}")
        # Output must be raw score. This rule must be followed.
        output = self.predict_func(input, *args, iteration_at=iteration_at, **kwargs)
        if hasattr(self, "inference") and self.inference is not None:
            LOGGER.info("extra inference for output...")
            output = self.inference(output)
        if is_softmax is None:
            is_softmax = self.is_softmax
        if is_softmax:
            LOGGER.info("softmax output...")
            if len(output.shape) == 1:
                output = sigmoid(output)
            else:
                output = softmax(output)
        LOGGER.info("END")
        return output
    def predict_xgb(self, input: np.ndarray, *args, iteration_at: int | None=None, **kwargs) -> np.ndarray:
        LOGGER.info("START")
        assert iteration_at is None or (isinstance(iteration_at, int) and iteration_at >= 0)
        if iteration_at is None: iteration_at = self.best_iteration
        output = self.booster.predict(xgb.DMatrix(input), *args, output_margin=True, iteration_range=(0, iteration_at), **kwargs)
        LOGGER.info("END")
        return output
    def predict_lgb(self, input: np.ndarray, *args, iteration_at: int | None=None, **kwargs) -> np.ndarray:
        LOGGER.info("START")
        assert iteration_at is None or (isinstance(iteration_at, int) and iteration_at >= 0)
        if iteration_at is None:
            iteration_at = self.best_iteration
            if iteration_at == 0: iteration_at = -1
        # Output with raw score is required. This rule must be followed.
        output = self.booster.predict(input, *args, num_iteration=iteration_at, raw_score=True, **kwargs)
        LOGGER.info("END")
        return output
    def predict_cat(self, input: np.ndarray, *args, iteration_at: int | None=None, **kwargs) -> np.ndarray:
        LOGGER.info("START")
        assert iteration_at is None or (isinstance(iteration_at, int) and iteration_at >= 0)
        assert "prediction_type" not in kwargs, f"{kwargs}"
        if iteration_at is None:
            iteration_at = self.best_iteration
            if iteration_at is None or iteration_at < 0:
                iteration_at = self.booster.tree_count_ - 1
        kwargs["prediction_type"] = "RawFormulaVal"
        kwargs["ntree_end"]       = kwargs.get("ntree_end", iteration_at + 1)
        output = self.booster.predict(input, *args, **kwargs)
        output = np.array(output)
        if output.ndim == 2 and output.shape[1] == 1:
            output = output.reshape(-1)
        LOGGER.info("END")
        return output
    def to_dict(self) -> dict:
        if self.mode == "xgb":
            str_model = base64.b64encode(self.booster.save_raw()).decode('ascii')
        elif self.mode == "lgb":
            str_model = self.booster.model_to_string()
        else:
            str_model = base64.b64encode(self.booster._serialize_model()).decode("ascii")
        return {
            "mode": self.mode,
            "classes_": self.classes_,
            "loss": self.loss.to_dict() if isinstance(self.loss, Loss) else str(self.loss),
            "params": self.params,
            "params_pkg": self.params_pkg,
            "is_softmax": self.is_softmax,
            "best_iteration": self.best_iteration,
            "total_iteration": self.total_iteration,
            "time_train": self.time_train,
            "feature_importances": self.feature_importances,
            "model": str_model, # long text goes last
        }
    def to_json(self, indent: int=None):
        return json.dumps(self.to_dict(), indent=indent)
    @classmethod
    def from_dict(cls, dict_model: dict):
        LOGGER.info("START")
        assert isinstance(dict_model, dict)
        if dict_model["mode"] == "xgb":
            booster = xgb.Booster(model_file=bytearray(base64.b64decode(dict_model["model"])))
        elif dict_model["mode"] == "lgb":
            booster = lgb.Booster(model_str=dict_model["model"])
        else:
            booster = cat.CatBoost().load_model(blob=bytes(bytearray(base64.b64decode(dict_model["model"]))))
        num_class = len(dict_model.get("classes_", [])) if len(dict_model.get("classes_", [])) > 0 else 1
        ins = cls(num_class, mode=dict_model["mode"])
        ins.booster = booster
        for x, y in dict_model.items():
            if x in ["model", "loss"]: continue
            setattr(ins, x, y)
        if isinstance(dict_model["loss"], dict):
            ins.loss = Loss.from_dict(dict_model["loss"])
        else:
            ins.loss = dict_model["loss"]
        if ins.is_softmax is None:
            # temporary code for backward compatibility
            ins.is_softmax = is_prob_from_loss_func(ins.loss, ins.mode)
        LOGGER.info("END")
        return ins
    @classmethod
    def from_json(cls, json_model: str):
        LOGGER.info("START")
        assert isinstance(json_model, str)
        ins = cls.from_dict(json.loads(json_model))
        LOGGER.info("END")
        return ins
    def dump_with_loader(self) -> dict:
        return {
            "__class__": "kkgbdt.model.KkGBDT",
            "__loader__": "from_json",
            "__dump_string__": self.to_json(indent=4),
        }


def _train_xgb(p: ParamsTraining, evals_result: dict = None):
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
    LOGGER.info("START")
    # check parameters can be done pre-process
    params = copy.deepcopy(p.params)
    x_train, y_train, x_valid, y_valid, group_train, group_valid = p.x_train, p.y_train, p.x_valid, p.y_valid, p.group_train, p.group_valid
    categorical_features = p.categorical_features
    loss_func, loss_func_eval = p.loss_func, p.loss_func_eval
    early_stopping_rounds, early_stopping_idx = p.early_stopping_rounds, p.early_stopping_idx
    train_stopping_val, train_stopping_time, train_stopping_rounds, train_stopping_is_over = \
        p.train_stopping_val, p.train_stopping_time, p.train_stopping_rounds, p.train_stopping_is_over
    # check objective & metric
    if params.get("objective")   is not None: LOGGER.raise_error(f"Don't set params['objective']   parameter, set 'loss_func' instead.")
    if params.get("eval_metric") is not None: LOGGER.raise_error(f"Don't set params['eval_metric'] parameter, set 'loss_func_eval' instead.")
    # check GPU
    if params.get("device") is not None and params["device"] == "cuda":
        # https://xgboost.readthedocs.io/en/stable/gpu/index.html
        LOGGER.info("Training with GPU mode.", color=["BOLD", "CYAN"])
        dataset_func = partial(create_xgb_dataset_with_group, use_quantile=True, max_bin=params["max_bin"])
    else:
        dataset_func = create_xgb_dataset_with_group
    # dataset option
    enable_categorical = False
    if categorical_features is not None:
        assert check_type_list(categorical_features, int)
        if len(categorical_features) > 0:
            _categorical_features = np.array(["q"] * x_train.shape[-1]) # "q" is numerical.
            _categorical_features[categorical_features] = "c"
            categorical_features = _categorical_features.tolist()
            enable_categorical   = True
        else:
            categorical_features = None
    # multi strategy
    if "multi_strategy" in params and params["multi_strategy"] == "multi_output_tree":
        # see: https://xgboost.readthedocs.io/en/stable/python/examples/multioutput_regression.html#sphx-glr-python-examples-multioutput-regression-py
        assert len(y_train.shape) == 2 and y_train.shape[-1] == params["num_class"]
        params["num_target"] = y_train.shape[-1]
        LOGGER.warning(
            f"Multi class and multi_strategy cannot use at the same time. " + 
            f"num_class: {params['num_class']}, num_target: {params['num_target']}. set num_class=1."
        )
        params["num_class"] = 1
    else:
        if len(y_train.shape) != 1:
            LOGGER.warning("XGBoost basically doesn't support multi task. You should use multi_output_tree.")
        assert len(y_train.shape) == 1
    # set dataset
    dataset_train = create_xgb_dataset_with_group(
        x_train, label=y_train, weight=p.sample_weight, group=group_train,
        feature_types=categorical_features, enable_categorical=enable_categorical
    )
    dataset_valid = [(dataset_train, "train")] + [
        (create_xgb_dataset_with_group(_x_valid, label=_y_valid, group=_group_valid), f"valid_{i_valid}")
        for i_valid, (_x_valid, _y_valid, _group_valid) in enumerate(zip(x_valid, y_valid, group_valid))
    ]
    # loss setting
    custom_loss_func, custom_loss_func_eval = None, None
    params["eval_metric"] = []
    params["disable_default_eval_metric"] = 1
    if isinstance(loss_func, str):
        params["objective"] = loss_func
    else:
        custom_loss_func = LGBCustomObjective(loss_func, mode="xgb")
    if isinstance(loss_func_eval, list) and len(loss_func_eval) > 0:
        for func_eval in loss_func_eval:
            if isinstance(func_eval, str):
                params["eval_metric"].append(func_eval)
            else:
                if custom_loss_func_eval is None: custom_loss_func_eval = []
                custom_loss_func_eval.append(LGBCustomEval(func_eval, mode="xgb"))
    else:
        # xgboost must have at least one eval metric if you want to get logs in callbacks
        if isinstance(loss_func, str):
            params["eval_metric"].append(str_loss_to_metric(loss_func, "xgb"))
        else:
            custom_loss_func_eval = [LGBCustomEval(loss_func, mode="xgb"), ]
    if custom_loss_func_eval is not None:
        if len(custom_loss_func_eval) > 1:
            LOGGER.warning(f"xgboost's custom metric is supported only one function. so set this metric: {custom_loss_func_eval[0]}")
        custom_loss_func_eval = custom_loss_func_eval[0]
    # callbacks
    callbacks = [PrintEvalation()]
    ## early stopping
    if early_stopping_rounds is not None:
        metric_name = check_early_stopping(early_stopping_rounds, early_stopping_idx, "xgb", loss_func_eval=loss_func_eval)
        callbacks.append(
            EarlyStopping(early_stopping_rounds, metric_name=metric_name, data_name="valid_0", save_best=True)
        )
    ## train stopping ( triggered by eval value is too low or training step time is too slow )
    if train_stopping_val is not None or train_stopping_time is not None:
        check_train_stopping(train_stopping_val, train_stopping_time, train_stopping_rounds, train_stopping_is_over)
        callbacks.append(
            TrainStopping(
                train_stopping_val=train_stopping_val, train_stopping_rounds=train_stopping_rounds, 
                train_stopping_is_over=train_stopping_is_over, train_stopping_time=train_stopping_time
            )
        )
    # train
    LOGGER.info(f"params: {params}")
    model = xgb.train(
        params, dataset_train, 
        num_boost_round=p.num_iterations, evals=dataset_valid, obj=custom_loss_func,
        custom_metric=custom_loss_func_eval, maximize=False, early_stopping_rounds=None,
        evals_result=evals_result, verbose_eval=False, xgb_model=None,
        callbacks=callbacks
    )
    LOGGER.info("END")
    return model


def _train_lgb(p: ParamsTraining, evals_result: dict = None, save_dataset: str | None=None):
    """
    Params::
        loss_func: custom loss or string
            string:
                see: https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective
                binary, multiclass, regression, ...
        loss_func_eval:
            custom loss or string.
            string:
                see: https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric
                rmse, auc, multi_logloss, ...
    """
    LOGGER.info("START")
    # check parameters can be done pre-process
    params = copy.deepcopy(p.params)
    x_train, y_train, x_valid, y_valid, group_train, group_valid = p.x_train, p.y_train, p.x_valid, p.y_valid, p.group_train, p.group_valid
    categorical_features = p.categorical_features
    loss_func, loss_func_eval = p.loss_func, p.loss_func_eval
    early_stopping_rounds, early_stopping_idx = p.early_stopping_rounds, p.early_stopping_idx
    train_stopping_val, train_stopping_time, train_stopping_rounds, train_stopping_is_over = \
        p.train_stopping_val, p.train_stopping_time, p.train_stopping_rounds, p.train_stopping_is_over
    # check objective & metric
    if params.get("objective") is not None: LOGGER.raise_error(f"please set 'objective' parameter to 'loss_func'.")
    if params.get("metric")    is not None: LOGGER.raise_error(f"please set 'metric'    parameter to 'loss_func_eval'.")
    # dataset option
    if categorical_features is not None:
        assert check_type_list(categorical_features, int)
    else:
        categorical_features = "auto"
    # set dataset
    if isinstance(x_train, str):
        dataset_train = DatasetLGB(x_train)
    else:
        dataset_train = DatasetLGB(
            x_train, label=y_train, weight=p.sample_weight, group=group_train, encode_type=p.encode_type,
            categorical_feature=categorical_features, params={"verbosity": params["verbosity"]}
        )
    if len(x_valid) > 0 and all(isinstance(x, str) for x in x_valid):
        dataset_valid = [dataset_train] + [DatasetLGB(_x_valid) for _x_valid in x_valid]
    else:
        dataset_valid = [dataset_train] + [
            DatasetLGB(
                _x_valid, label=_y_valid, reference=dataset_train, group=_group_valid, 
                encode_type=p.encode_type, params={"verbosity": params["verbosity"]}
            ) for _x_valid, _y_valid, _group_valid in zip(x_valid, y_valid, group_valid)
        ]
    # loss setting
    custom_loss_func_eval = None
    params["metric"] = []
    if isinstance(loss_func, str):
        params["objective"] = loss_func
    else:
        params["objective"] = LGBCustomObjective(loss_func, mode="lgb")
    if loss_func_eval is not None:
        for func_eval in loss_func_eval:
            if isinstance(func_eval, str):
                params["metric"].append(func_eval)
            else:
                if custom_loss_func_eval is None: custom_loss_func_eval = []
                custom_loss_func_eval.append(LGBCustomEval(func_eval, mode="lgb", is_higher_better=func_eval.is_higher_better))
    else:
        if not isinstance(loss_func, str):
            custom_loss_func_eval = [LGBCustomEval(loss_func, mode="lgb", is_higher_better=loss_func.is_higher_better), ]
    # callbacks
    callbacks = [
        record_evaluation(evals_result),
        log_evaluation(),
    ]
    ## early stopping
    dict_train, dict_eval_best = {}, {}
    if early_stopping_rounds is not None:
        early_stopping_idx = check_early_stopping(early_stopping_rounds, early_stopping_idx, "lgb", loss_func_eval=loss_func_eval)
        callbacks.append(callback_best_iter(dict_eval_best, early_stopping_rounds, early_stopping_idx))
    ## train stopping ( triggered by eval value is too low or training step time is too slow ) 
    if train_stopping_val is not None or train_stopping_time is not None:
        check_train_stopping(train_stopping_val, train_stopping_time, train_stopping_rounds, train_stopping_is_over)
        callbacks.append(
            callback_stop_training(dict_train, train_stopping_val, train_stopping_rounds, train_stopping_is_over, train_stopping_time)
        )
    # train
    LOGGER.info(f"params: {params}")
    model = lgb.train(
        params, dataset_train, num_boost_round=p.num_iterations,
        valid_sets=dataset_valid, valid_names=["train"]+[f"valid_{i}" for i in range(len(dataset_valid)-1)],
        feval=custom_loss_func_eval, callbacks=callbacks
    )
    if save_dataset is not None:
        LOGGER.info(f"save dataset to {save_dataset}.xxxxx.bin")
        for ds, name in zip(
            ([dataset_train, ] + dataset_valid[1:]),
            (["train", ] + [f"valid_{i}" for i in range(len(dataset_valid)-1)])
        ):
            bin_path = f"{save_dataset}.{name}.bin"
            if os.path.exists(bin_path):
                LOGGER.warning(f"File already exists, deleting: {bin_path}")
                os.remove(bin_path)
            ds.save_binary(bin_path)
    LOGGER.info("END")
    return model


def _train_cat(p: ParamsTraining, evals_result: dict = None):
    LOGGER.info("START")
    # check parameters can be done pre-process
    params = copy.deepcopy(p.params)
    x_train, y_train, x_valid, y_valid, group_train, group_valid = p.x_train, p.y_train, p.x_valid, p.y_valid, p.group_train, p.group_valid
    categorical_features = p.categorical_features
    loss_func, loss_func_eval = p.loss_func, p.loss_func_eval
    early_stopping_rounds, early_stopping_idx = p.early_stopping_rounds, p.early_stopping_idx
    train_stopping_val, train_stopping_time, train_stopping_rounds, train_stopping_is_over = \
        p.train_stopping_val, p.train_stopping_time, p.train_stopping_rounds, p.train_stopping_is_over
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds
        if early_stopping_idx is not None:
            LOGGER.warning("early_stopping_idx is not supported in cat mode, ignored. Please set a metric you want to use for early stopping to first position")
    if len(x_valid) == 0:
        params["use_best_model"] = False
    if categorical_features is not None:
        assert check_type_list(categorical_features, int)
    # train stopping is not supported in catboost
    if train_stopping_val is not None or train_stopping_time is not None:
        LOGGER.warning("train_stopping_* is not supported in cat mode, ignored.")
    # set dataset
    train_pool = DatasetCB(
        x_train, label=y_train, weight=p.sample_weight, group_id=group_train, 
        cat_features=categorical_features
    )
    valid_pools = [
        DatasetCB(_x_valid, label=_y_valid, group_id=_group_valid, cat_features=categorical_features)
        for _x_valid, _y_valid, _group_valid in zip(x_valid, y_valid, group_valid)
    ]
    # loss setting
    params["eval_metric"]   = None
    params["custom_metric"] = []
    if isinstance(loss_func, str):
        params["loss_function"] = loss_func
    else:
        params["loss_function"] = LGBCustomObjective(loss_func, mode="cat")
    if loss_func_eval is not None:
        for i, func_eval in enumerate(loss_func_eval):
            if i == 0:
                if isinstance(func_eval, str):
                    params["eval_metric"] = func_eval
                else:
                    params["eval_metric"] = LGBCustomEval(func_eval, mode="cat", is_higher_better=func_eval.is_higher_better)
            else:
                if isinstance(func_eval, str):
                    params["custom_metric"].append(func_eval)
                else:
                    LOGGER.raise_error(f"custom metric is not supported in 'catboost/custom_metric' parameter. This means that only one custom metric is valid.")
    # callbacks
    params['callbacks'] = create_callbacks_cb()
    # train
    if params["classes_count"] == 1: params["classes_count"] = None
    params = {x: y for x, y in params.items() if y is not None}
    LOGGER.info(f"params: {params}")
    model = cat.train(
        pool=train_pool, params=params, iterations=p.num_iterations, eval_set=valid_pools
    )
    # evals result
    evals_result.clear()
    dict_result = model.get_evals_result()
    for data_name, metrics in dict_result.items():
        if data_name == "learn":
            key = "train"
        elif data_name.startswith("validation_"):
            key = data_name.replace("validation_", "valid_")
        else:
            key = data_name
        evals_result[key] = metrics
    LOGGER.info("END")
    return model


def set_random_seed(mode: str, random_seed: int) -> dict[str, int]:
    check_mode(mode)
    assert isinstance(random_seed, int) and random_seed >= 0
    return {
        "xgb": {"seed": random_seed},
        "lgb": {
            "seed": random_seed,
            "feature_fraction_seed": random_seed, # for feature_fraction
            "extra_seed": random_seed, # for extra_trees
        },
        "cat": {"random_seed": random_seed},
    }[mode]


def alias_parameters(
    mode: str="xgb",
    num_class: int=None, learning_rate: float=None, n_jobs: int=None, is_gpu: bool=None, 
    max_bin: int=None, random_seed: int=0, max_depth: int | None=None, num_leaves: int | None=None, 
    min_child_samples: int | None=None, min_child_weight: float | None=None, subsample: float | None=None,
    colsample_bytree: float | None=None, colsample_bylevel: float | None=None, colsample_bynode: float | None=None,
    reg_alpha: float | None=None, reg_lambda: float | None=None,
    min_split_gain: float | None=None, min_data_in_bin: int | None=None, path_smooth: float | None=None, 
    multi_strategy: str | None=None, grow_policy: str | None=None, is_extratree: bool | None=None, verbosity: int | None=None,
) -> dict[str, Any]:
    LOGGER.info(f"START. mode={mode}")
    check_mode(mode)
    assert isinstance(num_class, int) and num_class > 0
    assert isinstance(learning_rate, float) and learning_rate >= 1e-5 and learning_rate <= 1
    assert isinstance(n_jobs, int) and (n_jobs > 0 or n_jobs == -1)
    assert isinstance(is_gpu, bool)
    assert isinstance(max_bin, int) and max_bin >= 4
    assert max_depth         is None or (isinstance(max_depth, int) and max_depth >= -1)
    assert num_leaves        is None or (isinstance(num_leaves, int) and num_leaves >= 0)
    assert min_child_samples is None or (isinstance(min_child_samples, int) and min_child_samples > 0)
    assert min_child_weight  is None or (check_type(min_child_weight, [float, int]) and min_child_weight >= 1e-5)
    assert subsample         is None or (check_type(subsample, [float, int]) and subsample > 0 and subsample <= 1)
    assert colsample_bytree  is None or (check_type(colsample_bytree,  [float, int]) and colsample_bytree  > 0 and colsample_bytree  <= 1)
    assert colsample_bylevel is None or (check_type(colsample_bylevel, [float, int]) and colsample_bylevel > 0 and colsample_bylevel <= 1)
    assert colsample_bynode  is None or (check_type(colsample_bynode,  [float, int]) and colsample_bynode  > 0 and colsample_bynode  <= 1)
    assert reg_alpha         is None or (check_type(reg_alpha,  [float, int]) and reg_alpha  >= 0)
    assert reg_lambda        is None or (check_type(reg_lambda, [float, int]) and reg_lambda >= 0)
    assert min_split_gain    is None or (check_type(min_split_gain, [float, int]) and min_split_gain >= 0)
    assert min_data_in_bin   is None or (isinstance(min_data_in_bin, int) and min_data_in_bin >= 1)
    assert path_smooth       is None or (check_type(path_smooth, [int, float]) and path_smooth >= 0)
    assert multi_strategy    is None or (multi_strategy in ["one_output_per_tree", "multi_output_tree"])
    assert grow_policy       is None or (isinstance(grow_policy, str) and grow_policy in ["depthwise", "lossguide", "symmetric"])
    assert is_extratree      is None or (isinstance(is_extratree, bool))
    assert verbosity         is None or (isinstance(verbosity, int) and verbosity >= 0)
    params = {}
    if mode == "xgb":
        params["num_class"]   = num_class
        params["eta"]         = learning_rate
        params["max_leaves"]  = num_leaves
        params["nthread"]     = n_jobs
        params["booster"]     = "gbtree" # "dart", "gblinear"
        params["tree_method"] = "hist"
        params["device"]      = "cuda" if is_gpu else "cpu"
        params["max_depth"]   = 0 if max_depth <= 0 else max_depth
        params["min_child_weight"]  = min_child_weight
        params["subsample"]         = subsample
        params["colsample_bytree"]  = colsample_bytree
        params["colsample_bylevel"] = colsample_bylevel
        params["colsample_bynode"]  = colsample_bynode
        params["alpha"]             = reg_alpha
        params["lambda"]            = reg_lambda
        params["gamma"]             = min_split_gain # Maybe different from xgb and lgb
        params["max_bin"]           = max_bin
        params["verbosity"]         = 0 if verbosity is None else verbosity
        params["multi_strategy"]    = "one_output_per_tree" if multi_strategy is None else multi_strategy
        assert grow_policy is None or (grow_policy in ["depthwise", "lossguide"])
        params["grow_policy"]       = "depthwise" if grow_policy is None else grow_policy
        for x in ["min_child_samples", "min_data_in_bin", "path_smooth", "is_extratree"]:
            if locals()[x] is not None: LOGGER.warning(f"{x} is not in configlation for {mode}")
    elif mode == "lgb":
        params["num_class"]               = num_class
        params["learning_rate"]           = learning_rate
        params["num_leaves"]              = num_leaves
        params["num_threads"]             = n_jobs
        params["device_type"]             = "cpu" if is_gpu is None or is_gpu == False else "gpu"
        if is_gpu is not None and is_gpu:
            params["gpu_platform_id"]     = 0
            params["gpu_device_id"]       = 0
        params["extra_trees"]             = is_extratree
        params["max_depth"]               = -1 if max_depth <= 0 else max_depth
        params["min_data_in_leaf"]        = min_child_samples
        params["min_sum_hessian_in_leaf"] = min_child_weight
        params["bagging_fraction"]        = subsample
        if subsample is not None and subsample != 1:
            params["bagging_freq"]        = 1
        params["feature_fraction"]        = colsample_bytree
        params["feature_fraction_bynode"] = colsample_bynode
        params["lambda_l1"]               = reg_alpha
        params["lambda_l2"]               = reg_lambda
        params["min_gain_to_split"]       = min_split_gain
        params["max_bin"]                 = max_bin - 1
        params["min_data_in_bin"]         = min_data_in_bin
        params["path_smooth"]             = path_smooth
        params["booster"]                 = "gbdt" # "rf", "dart"
        params["deterministic"]           = False # https://lightgbm.readthedocs.io/en/stable/Parameters.html#deterministic
        if path_smooth is not None and path_smooth > 0.0:
            assert params["min_data_in_leaf"] >= 2
        params["verbosity"]               = -1 if verbosity is None else verbosity
        for x in ["colsample_bylevel", "multi_strategy", "grow_policy"]:
            if locals()[x] is not None: LOGGER.warning(f"{x} is not in configlation for {mode}")
    else:
        params["classes_count"]       = num_class
        params["learning_rate"]       = learning_rate
        params["max_leaves"]          = num_leaves
        params["thread_count"]        = n_jobs
        params["task_type"]           = "GPU" if is_gpu else "CPU"
        params["depth"]               = max_depth
        params["min_data_in_leaf"]    = min_child_samples
        params["subsample"]           = None if subsample == 1 else subsample
        # colsample_bytree, colsample_bynode is not supported in catboost
        # colsample_bytree:  決定木毎に特徴量をランダム選択して、その特徴量のみで木を作る
        # colsample_bylevel: 決定木の階層毎に特徴量をランダム選択して、その階層はその特徴量のみで作られる
        # colsample_bynode:  ノード毎に特徴量をランダム選択して、そのノードはその特徴量のみで作られる
        params["rsm"]                 = colsample_bylevel
        params["l2_leaf_reg"]         = reg_lambda # reg_alpha is not supported in catboost
        params["border_count"]        = max_bin
        params["allow_writing_files"] = False
        params["bootstrap_type"]      = "Bernoulli"
        if grow_policy is None:
            params["grow_policy"]     = "SymmetricTree"
        else:
            params["grow_policy"]     = {"depthwise": "Depthwise", "lossguide": "Lossguide", "symmetric": "SymmetricTree"}[grow_policy]
        params["use_best_model"]      = True
        params["od_type"]             = "Iter"
        if verbosity is None or verbosity <= 0:
            params["logging_level"]   = "Silent"
        elif verbosity == 1:
            params["logging_level"]   = "Info"
        else:
            params["logging_level"]   = "Verbose"
        for x in [
            "min_child_weight", "colsample_bytree", "colsample_bynode", "reg_alpha",
            "min_split_gain", "min_data_in_bin", "path_smooth", "multi_strategy", "is_extratree"
        ]:
            if locals()[x] is not None: LOGGER.warning(f"{x} is not in configlation for {mode}")
    params  = params | set_random_seed(mode, random_seed)
    _params = {}
    for x, y in params.items():
        if y is not None: _params[x] = y
    LOGGER.info("END")
    return _params


def set_all_loglevel(log_level: str):
    LOGGER.info("START")
    try: set_loglevel("kkgbdt.model",     log_level=log_level)
    except LoggingNameException: pass 
    try: set_loglevel("kkgbdt.dataset",   log_level=log_level)
    except LoggingNameException: pass 
    try: set_loglevel("kkgbdt.callbacks", log_level=log_level)
    except LoggingNameException: pass 
    try: set_loglevel("kkgbdt.loss",      log_level=log_level)
    except LoggingNameException: pass 
    try: set_loglevel("kkgbdt.trace",     log_level=log_level)
    except LoggingNameException: pass 
    try: set_loglevel("kkgbdt.tune",      log_level=log_level)
    except LoggingNameException: pass 
    LOGGER.info("END")

