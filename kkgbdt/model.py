import copy, time, json
import numpy as np
from functools import partial
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
from xgboost.callback import EarlyStopping
from lightgbm.callback import record_evaluation
from kkgbdt.loss import Loss, LGBCustomObjective, LGBCustomEval
from kkgbdt.dataset import DatasetLGB
from kkgbdt.trace import KkTracer
from kkgbdt.callbacks import PrintEvalation, TrainStopping, log_evaluation, callback_stop_training, callback_best_iter
from kkgbdt.util.numpy import softmax
from kkgbdt.util.com import check_type, check_type_list
from kklogger import set_logger
logger = set_logger(__name__)


__all__ = [
    "KkGBDT",
    "train_xgb",
    "train_lgb",
    "alias_parameter",
]


class KkGBDT:
    def __init__(
        self, num_class: int, mode: str="xgb", is_softmax: bool=False, 
        learning_rate: float=0.1, num_leaves: int=100, n_jobs: int=-1, is_gpu: bool=False, 
        random_seed: int=0, max_depth: int=-1, min_child_samples: int=20, min_child_weight: float=1e-3,
        subsample: float=1, colsample_bytree: float=0.5, colsample_bylevel: float=1, colsample_bynode: float=1,
        reg_alpha: float=0, reg_lambda: float=0, min_split_gain: float=0, max_bin: int=256, 
        min_data_in_bin: int=5, path_smooth: float=0.0, multi_strategy: str=None, verbosity: None | int=None, **kwargs
    ):
        logger.info("START")
        assert isinstance(mode, str) and mode in ["xgb", "lgb"]
        assert isinstance(is_softmax, bool)
        self.booster = None
        self.mode    = mode
        self.params  = alias_parameters(
            mode=self.mode, num_class=num_class, learning_rate=learning_rate, num_leaves=num_leaves, n_jobs=n_jobs, is_gpu=is_gpu, 
            random_seed=random_seed, max_depth=max_depth, min_child_samples=min_child_samples, min_child_weight=min_child_weight,
            subsample=subsample, colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode, reg_alpha=reg_alpha, reg_lambda=reg_lambda, min_split_gain=min_split_gain,
            max_bin=max_bin, min_data_in_bin=min_data_in_bin, path_smooth=path_smooth, multi_strategy=multi_strategy, verbosity=verbosity
        )
        self.params.update(copy.deepcopy(kwargs))
        self.params_pkg      = {}
        self.evals_result    = {}
        self.classes_        = np.arange(num_class, dtype=int)
        self.is_softmax      = is_softmax
        self.time_train      = 0
        self.best_iteration  = 0
        self.total_iteration = 0
        self.loss            = None
        self.inference       = None
        if mode == "xgb":
            self.train_func   = train_xgb
            self.predict_func = self.predict_xgb
        else:
            self.train_func   = train_lgb
            self.predict_func = self.predict_lgb
        logger.info(f"params: {self.params}")
        logger.info("END")
    def to_json(self):
        return {
            "mode": self.mode,
            "classes": self.classes_.tolist(),
            "loss": str(self.loss),
            "params": self.params,
            "params_pkg": self.params_pkg,
            "is_softmax": self.is_softmax,
            "best_iteration": self.best_iteration,
            "total_iteration": self.total_iteration,
            "time_train": self.time_train,
        }
    def __str__ (self):
        return self.__class__.__name__ + " " + json.dumps(self.to_json(), indent=4)
    def copy(self):
        instance = copy.deepcopy(self)
        for x in ["best_iteration", "best_score"]:
            if hasattr(self.booster, x):
                setattr(instance.booster, x, copy.deepcopy(getattr(self.booster, x)))
        return instance
    def fit(
        self, x_train: np.ndarray, y_train: np.ndarray, loss_func: str | Loss=None, num_iterations: int=None,
        x_valid: np.ndarray | list[np.ndarray]=None, y_valid: np.ndarray | list[np.ndarray]=None,
        loss_func_eval: str | Loss=None, early_stopping_rounds: int=None, early_stopping_name: int | str=None,
        stopping_name: str=None, stopping_val: float=None, stopping_rounds: int=None, stopping_is_over: bool=True, stopping_train_time: float=None,
        sample_weight: str | np.ndarray | list[str | np.ndarray]=None, categorical_features: list[int]=None
    ):
        logger.info("START")
        assert loss_func is not None
        assert num_iterations is not None
        self.loss = loss_func
        if isinstance(self.loss, Loss) and hasattr(self.loss, "inference"):
            self.inference = self.loss.inference
        # common processing
        if loss_func_eval is not None:
            if not isinstance(loss_func_eval, list): loss_func_eval = [loss_func_eval, ]
            _loss_func_eval = []
            for func_eval in loss_func_eval:
                if isinstance(func_eval, str) and func_eval == "__copy__":
                    assert isinstance(loss_func, Loss)
                    _loss_func_eval.append(loss_func)
                else:
                    _loss_func_eval.append(func_eval)
            loss_func_eval = _loss_func_eval
        if sample_weight is not None:
            # sample_weight can be multi. calculate step by step
            if isinstance(sample_weight, str) or isinstance(sample_weight, np.ndarray): sample_weight = [sample_weight, ]
            assert check_type_list(sample_weight, [str, np.ndarray])
            _sample_weight = np.ones(y_train.shape[0]).astype(float)
            logger.info(f"sample weight before: {sample_weight}")
            for sw in sample_weight:
                if isinstance(sw, str):
                    assert sw in ["balanced"]
                    assert len(y_train.shape) == 1 and y_train.dtype in [int, np.int16, np.int32, np.int64]
                    assert np.unique(y_train).shape == np.bincount(y_train).shape
                    _sample_weight = _sample_weight * compute_sample_weight("balanced", y_train) # https://github.com/microsoft/LightGBM/blob/a5285985992ebd376c356a0ac1d10a190338550b/python-package/lightgbm/sklearn.py#L771
                else:
                    assert len(sw.shape) == 1 and len(y_train.shape) == 1 and y_train.shape[0] == sw.shape[0]
                    _sample_weight = _sample_weight * sw
            sample_weight = _sample_weight.copy()
            logger.info(f"sample weight after:  {sample_weight}")
        # training
        time_start = time.time()
        self.booster = self.train_func(
            copy.deepcopy(self.params), num_iterations, x_train, y_train, self.loss,
            evals_result=self.evals_result, x_valid=x_valid, y_valid=y_valid, loss_func_eval=loss_func_eval,
            early_stopping_rounds=early_stopping_rounds, early_stopping_name=early_stopping_name,
            stopping_name=stopping_name, stopping_val=stopping_val, stopping_rounds=stopping_rounds, 
            stopping_is_over=stopping_is_over, stopping_train_time=stopping_train_time,
            sample_weight=sample_weight, categorical_features=categorical_features,
        )
        self.time_train = time.time() - time_start
        # save params
        if self.mode == "xgb":
            self.params_pkg = json.loads(self.booster.save_config())
        else:
            self.params_pkg = copy.deepcopy(self.booster.params)
        # best iteration
        if self.evals_result.get("valid_0") is not None:
            if early_stopping_name is None: early_stopping_name = 0
            if isinstance(early_stopping_name, int):
                list_vals = list(self.evals_result.get("valid_0").values())[early_stopping_name]
            else:
                list_vals = self.evals_result.get("valid_0")[early_stopping_name]
            self.best_iteration  = int(np.argmin(np.array(list_vals)))
            self.total_iteration = len(list_vals)
            logger.info(f"best iteration is {self.best_iteration}.")
        if self.mode == "lgb": self.booster.__class__ = KkTracer
        # additional prosessing for custom loss
        if isinstance(self.loss, Loss) and hasattr(self.loss, "extra_processing"):
            output = self.predict_func(x_train)
            self.loss.extra_processing(output, y_train)
        self.set_parameter_after_training()
        logger.info(f"Time: {self.time_train} s.")
        logger.info("END")
    def set_parameter_after_training(self):
        if self.mode == "xgb":
            self.feature_importances_ = self.booster.get_fscore()
        else:
            self.feature_importances_ = self.booster.feature_importance()
    def predict(self, input: np.ndarray, *args, is_softmax: bool=None, iteration_at: int=None, **kwargs):
        logger.info(f"args: {args}, is_softmax: {is_softmax}, kwargs: {kwargs}")
        output = self.predict_func(input, *args, iteration_at=iteration_at, **kwargs)
        if hasattr(self, "inference") and self.inference is not None:
            logger.info("extra inference for output...")
            output = self.inference(output)
        if is_softmax is None: is_softmax = self.is_softmax
        if is_softmax:
            logger.info("softmax output...")
            output = softmax(output)
        logger.info("END")
        return output
    def predict_xgb(self, input: np.ndarray, *args, iteration_at: int=None, **kwargs):
        logger.info("START")
        assert iteration_at is None or (isinstance(iteration_at, int) and iteration_at >= 0)
        if iteration_at is None: iteration_at = self.best_iteration
        output = self.booster.predict(xgb.DMatrix(input), *args, output_margin=True, iteration_range=(0, iteration_at), **kwargs)
        logger.info("END")
        return output
    def predict_lgb(self, input: np.ndarray, *args, iteration_at: int=None, **kwargs):
        logger.info("START")
        assert iteration_at is None or (isinstance(iteration_at, int) and iteration_at >= 0)
        if iteration_at is None:
            iteration_at = self.best_iteration
            if iteration_at == 0: iteration_at = -1
        output = self.booster.predict(input, *args, num_iteration=iteration_at, **kwargs)
        logger.info("END")
        return output


def train_xgb(
    # fitting parameter
    params: dict, num_iterations: int,
    # training data & loss
    x_train: np.ndarray, y_train: np.ndarray, loss_func: str | Loss, evals_result: dict={}, 
    # validation data & loss
    x_valid: np.ndarray | list[np.ndarray]=None, y_valid: np.ndarray | list[np.ndarray]=None, loss_func_eval: str | Loss=None,
    # early stopping parameter
    early_stopping_rounds: int=None, early_stopping_name: int | str=None,
    stopping_name: str=None, stopping_val: float=None, stopping_rounds: int=None, stopping_is_over: bool=True, stopping_train_time: float=None,
    # option
    sample_weight: None | np.ndarray=None, categorical_features: list[int]=None
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
    assert isinstance(num_iterations, int) and num_iterations > 0
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
    # check GPU
    if params.get("device") is not None and params["device"] == "cuda":
        # https://xgboost.readthedocs.io/en/stable/gpu/index.html
        logger.info("Training with GPU mode.")
        dataset_class = partial(xgb.QuantileDMatrix, max_bin=params["max_bin"])
    else:
        dataset_class = xgb.DMatrix
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
    if params.get("multi_strategy") is not None and params["multi_strategy"] == "multi_output_tree":
        # see: https://xgboost.readthedocs.io/en/stable/python/examples/multioutput_regression.html#sphx-glr-python-examples-multioutput-regression-py
        assert len(y_train.shape) == 2
        params["num_target"] = y_train.shape[-1]
        if params["num_class"] != 1:
            logger.raise_error(f"You cannot use multi class and multi_strategy at the same time. num_class: {params['num_class']}, num_target: {params['num_target']}")
    else:
        if len(y_train.shape) != 1:
            logger.warning("XGBoost basically doesn't support multi task. You should use multi_output_tree.")
        assert len(y_train.shape) == 1
    # set dataset
    dataset_train = dataset_class(x_train, label=y_train, weight=sample_weight, feature_types=categorical_features, enable_categorical=enable_categorical)
    dataset_valid = [(dataset_train, "train")] + [(dataset_class(_x_valid, label=_y_valid), f"valid_{i_valid}") for i_valid, (_x_valid, _y_valid) in enumerate(zip(x_valid, y_valid))]
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
    logger.info(f"params: {params}")
    model = xgb.train(
        params, dataset_train, 
        num_boost_round=num_iterations, evals=dataset_valid, obj=_loss_func,
        custom_metric=_loss_func_eval, maximize=False, early_stopping_rounds=None,
        evals_result=evals_result, verbose_eval=False, xgb_model=None,
        callbacks=callbacks
    )
    logger.info("END")
    return model


def train_lgb(
    # fitting parameter
    params: dict, num_iterations: int,
    # training data & loss
    x_train: np.ndarray, y_train: np.ndarray, loss_func: str | Loss, evals_result: dict={}, 
    # validation data & loss
    x_valid: np.ndarray | list[np.ndarray]=None, y_valid: np.ndarray | list[np.ndarray]=None, loss_func_eval: str | Loss | list[str | Loss]=None,
    # early stopping parameter
    early_stopping_rounds: int=None, early_stopping_name: int | str=None,
    stopping_name:str=None, stopping_val: float=None, stopping_rounds: int=None, stopping_is_over: bool=True, stopping_train_time: float=None,
    # option
    sample_weight: None | np.ndarray=None, categorical_features: list[int]=None
):
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
    logger.info("START")
    assert isinstance(params, dict)
    assert isinstance(x_train, np.ndarray) and len(x_train.shape) == 2
    assert isinstance(y_train, np.ndarray) and len(y_train.shape) in [1, 2]
    assert isinstance(loss_func, str) or isinstance(loss_func, Loss)
    assert isinstance(num_iterations, int) and num_iterations > 0
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
    if params.get("objective") is not None: logger.raise_error(f"please set 'objective' parameter to 'loss_func'.")
    if params.get("metric")    is not None: logger.raise_error(f"please set 'metric'    parameter to 'loss_func_eval'.")
    params["num_iterations"] = num_iterations
    # dataset option
    if categorical_features is not None:
        assert check_type_list(categorical_features, int)
    else:
        categorical_features = "auto"
    # set dataset
    dataset_train = DatasetLGB(x_train, label=y_train, weight=sample_weight, categorical_feature=categorical_features, params={"verbosity": params["verbosity"]})
    dataset_valid = [dataset_train] + [DatasetLGB(_x_valid, label=_y_valid, params={"verbosity": params["verbosity"]}) for _x_valid, _y_valid in zip(x_valid, y_valid)]
    # loss setting
    _loss_func_eval = None
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
                if _loss_func_eval is None: _loss_func_eval = []
                _loss_func_eval.append(LGBCustomEval(func_eval, mode="lgb", is_higher_better=func_eval.is_higher_better))
    else:
        if isinstance(loss_func, str):
            _loss_func_eval = [loss_func, ]
        else:
            _loss_func_eval = [LGBCustomEval(loss_func, mode="lgb", is_higher_better=loss_func.is_higher_better), ]
    # callbacks
    callbacks = [
        record_evaluation(evals_result),
        log_evaluation(),
    ]
    ## early stopping
    dict_train, dict_eval_best = {}, {}
    if early_stopping_rounds is not None:
        assert isinstance(early_stopping_rounds, int) and early_stopping_rounds > 0
        assert check_type(early_stopping_name, [int, str])
        if isinstance(early_stopping_name, int):
            assert (early_stopping_name >= 0) and (loss_func_eval is not None) 
            metric_name = loss_func_eval[early_stopping_name]
            if isinstance(metric_name, Loss): metric_name = metric_name.name
            else:
                dict_conv = {
                    "multiclass": "multi_logloss",
                    "binary": "binary_logloss",
                }
                assert metric_name in dict_conv
                metric_name = dict_conv[metric_name]
        else:
            metric_name = early_stopping_name
        callbacks.append(callback_best_iter(dict_eval_best, early_stopping_rounds, metric_name))
    ## train stopping
    if stopping_val is not None or stopping_train_time is not None:
        assert stopping_val is None or isinstance(stopping_val, float)
        if stopping_train_time is not None:
            assert isinstance(stopping_train_time, float)
            assert isinstance(stopping_rounds, int) and stopping_rounds > 0
            assert isinstance(stopping_is_over, bool)
        callbacks.append(
            callback_stop_training(dict_train, stopping_val, stopping_rounds, stopping_is_over, stopping_train_time)
        )
    # train
    logger.info(f"params: {params}")
    model = lgb.train(
        params, dataset_train,
        valid_sets=dataset_valid, valid_names=["train"]+[f"valid_{i}" for i in range(len(dataset_valid)-1)],
        feval=_loss_func_eval, callbacks=callbacks
    )
    logger.info("END")
    return model


def alias_parameters(
    mode: str="xgb",
    num_class: int=None, learning_rate: float=None, num_leaves: int=None, n_jobs: int=None, is_gpu: bool=None, 
    random_seed: int=0, max_depth: int=None, min_child_samples: int=None, min_child_weight: float=None,
    subsample: float=None, colsample_bytree: float=None, colsample_bylevel: float=None,
    colsample_bynode: float=None, reg_alpha: float=None, reg_lambda: float=None, min_split_gain: float=None, 
    max_bin: int=None, min_data_in_bin: int=None, path_smooth: float=None, multi_strategy: str=None, verbosity: int=None,
):
    logger.info(f"START. mode={mode}")
    assert isinstance(mode, str) and mode in ["xgb", "lgb"]
    assert isinstance(num_class, int) and num_class > 0
    assert isinstance(learning_rate, float) and learning_rate >= 1e-5 and learning_rate <= 1
    assert isinstance(num_leaves, int) and num_leaves >= 0
    assert isinstance(n_jobs, int) and (n_jobs > 0 or n_jobs == -1)
    assert isinstance(is_gpu, bool)
    assert isinstance(random_seed, int) and random_seed >= 0
    assert isinstance(max_depth, int) and max_depth >= -1
    assert isinstance(min_child_samples, int) and min_child_samples > 0
    assert check_type(min_child_weight, [float, int]) and min_child_weight >= 1e-5
    assert check_type(subsample, [float, int]) and subsample > 0 and subsample <= 1
    assert check_type(colsample_bytree,  [float, int]) and colsample_bytree  > 0 and colsample_bytree  <= 1
    assert check_type(colsample_bylevel, [float, int]) and colsample_bylevel > 0 and colsample_bylevel <= 1
    assert check_type(colsample_bynode,  [float, int]) and colsample_bynode  > 0 and colsample_bynode  <= 1
    assert check_type(reg_alpha,  [float, int]) and reg_alpha >= 0
    assert check_type(reg_lambda, [float, int]) and reg_alpha >= 0
    assert check_type(min_split_gain, [float, int]) and min_split_gain >= 0
    assert isinstance(max_bin, int) and max_bin >= 4
    assert isinstance(min_data_in_bin, int) and min_data_in_bin >= 1
    assert check_type(path_smooth, [int, float]) and path_smooth >= 0
    assert multi_strategy is None or multi_strategy in ["one_output_per_tree", "multi_output_tree"]
    assert verbosity is None or isinstance(verbosity, int)
    params = {}
    if mode == "xgb":
        params["num_class"]   = num_class
        params["eta"]         = learning_rate
        params["max_leaves"]  = num_leaves
        params["nthread"]     = n_jobs
        params["tree_method"] = "hist"
        params["device"]      = "cuda" if is_gpu else "cpu"
        params["seed"]        = random_seed
        params["max_depth"]   = 0 if max_depth <= 0 else max_depth
        # min_child_samples is not in configlation
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
        params["grow_policy"]       = "depthwise"
        params["multi_strategy"]    = "one_output_per_tree" if multi_strategy is None else multi_strategy
        # min_data_in_bin is not in configlation
        for x in ["min_child_weight", "min_data_in_bin", "path_smooth"]:
            if locals()[x] is not None: logger.warning(f"{x} is not in configlation for {mode}")
    else:
        params["num_class"]               = num_class
        params["learning_rate"]           = learning_rate
        params["num_leaves"]              = num_leaves
        params["num_threads"]             = n_jobs
        params["device_type"]             = "cpu" if is_gpu is None or is_gpu == False else "gpu"
        if is_gpu is not None and is_gpu:
            params["gpu_platform_id"]     = 0
            params["gpu_device_id"]       = 0
        params["seed"]                    = random_seed
        params["max_depth"]               = -1 if max_depth <= 0 else max_depth
        params["min_data_in_leaf"]        = min_child_samples
        params["min_sum_hessian_in_leaf"] = min_child_weight
        params["bagging_fraction"]        = subsample
        if subsample is not None and subsample != 1:
            params["bagging_freq"]        = 1
        params["feature_fraction"]        = colsample_bytree
        # colsample_bylevel is not in configlation
        params["feature_fraction_bynode"] = colsample_bynode
        params["lambda_l1"]               = reg_alpha
        params["lambda_l2"]               = reg_lambda
        params["min_gain_to_split"]       = min_split_gain
        params["max_bin"]                 = max_bin - 1
        params["min_data_in_bin"]         = min_data_in_bin
        params["path_smooth"]             = path_smooth
        if path_smooth > 0.0: assert params["min_data_in_leaf"] >= 2
        params["verbosity"]               = -1 if verbosity is None else verbosity
        for x in ["colsample_bylevel", "multi_strategy"]:
            if locals()[x] is not None: logger.warning(f"{x} is not in configlation for {mode}")
    _params = {}
    for x, y in params.items():
        if y is not None: _params[x] = y
    logger.info("END")
    return _params
