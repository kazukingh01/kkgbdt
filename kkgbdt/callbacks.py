import time
from typing import List, Union
from xgboost.callback import TrainingCallback
from lightgbm.callback import EarlyStopException, _format_eval_result, _LogEvaluationCallback, CallbackEnv
from kkgbdt.util.com import check_type
from kklogger import set_logger


__all__ = [
    "PrintEvalation",
    "TrainStopping",
    "callback_stop_training",
    "callback_best_iter",
    "log_evaluation"
]


LOGGER = set_logger(__name__)


class PrintEvalation(TrainingCallback):
    def after_iteration(self, model, epoch: int, evals_log: TrainingCallback.EvalsLog) -> bool:
        string = f"[{epoch}]  "
        for name_data, dictwk in evals_log.items():
            for x, y in dictwk.items():
                string += f"{name_data}'s {x}: {str(y[-1])[:7]}\t"
        LOGGER.info(string)
        return False


class TrainStopping(TrainingCallback):
    def __init__(self, metric_name: str, stopping_val: float=None, stopping_rounds: int=None, stopping_is_over: bool=True, stopping_train_time: float=None):
        super().__init__()
        assert isinstance(metric_name, str)
        if stopping_val is not None:
            assert isinstance(stopping_val, float)
            assert isinstance(stopping_rounds, int) and stopping_rounds > 0
            assert isinstance(stopping_is_over, bool)
        if isinstance(stopping_train_time, int): stopping_train_time = float(stopping_train_time)
        if stopping_train_time is not None: assert isinstance(stopping_train_time, float) and stopping_train_time > 0
        self.metric_name         = metric_name
        self.stopping_val        = stopping_val
        self.stopping_rounds     = stopping_rounds
        self.stopping_is_over    = stopping_is_over
        self.stopping_train_time = stopping_train_time
        self.dict_train          = {"time": 9999999999}
    def after_iteration(self, model, epoch: int, evals_log: TrainingCallback.EvalsLog) -> bool:
        if (self.stopping_train_time is not None) and ((time.time() - self.dict_train["time"]) > self.stopping_train_time):
            LOGGER.info("Stopping training because time is over.")
            return True
        if self.stopping_val is not None:
            boolwk = (evals_log["train"][self.metric_name][-1] > self.stopping_val) if self.stopping_is_over else (evals_log["train"][self.metric_name][-1] < self.stopping_val)
            if (epoch > self.stopping_rounds) and boolwk:
                LOGGER.info(f"Stopping training because loss value is {'over' if self.stopping_is_over else 'less'} than {self.stopping_val}.")
                return True
        self.dict_train["time"] = time.time()
        return False


def callback_stop_training(dict_train: dict, stopping_val: float, stopping_rounds: int, stopping_is_over: bool, stopping_train_time: float):
    """
    If training loss does not reach the threshold, it will be terminated first.
    """
    def _init(env):
        dict_train["time"] = time.time()
    def _callback(env):
        if not dict_train: _init(env)
        _, _, result, _ = env.evaluation_result_list[0]
        if check_type(stopping_train_time, [int, float]) and ((time.time() - dict_train["time"]) > stopping_train_time):
            LOGGER.info(f'stop training. iteration: {env.iteration}, score: {result}')
            raise EarlyStopException(env.iteration, env.evaluation_result_list)
        if check_type(stopping_val, [float, int]) and isinstance(stopping_rounds, int) and env.iteration >= stopping_rounds:
            if stopping_is_over and result > stopping_val:
                LOGGER.info(f'stop training. iteration: {env.iteration}, score: {result}')
                raise EarlyStopException(env.iteration, env.evaluation_result_list)
            elif stopping_is_over == False and result < stopping_val:
                LOGGER.info(f'stop training. iteration: {env.iteration}, score: {result}')
                raise EarlyStopException(env.iteration, env.evaluation_result_list)
        dict_train["time"] = time.time()
    _callback.order = 150
    return _callback


def callback_best_iter(dict_eval: dict, stopping_rounds: int, name: Union[str, int]):
    """
    Determine best iteration for valid0
    """
    def _init(env):
        dict_eval["best_iter"]  = 0
        dict_eval["eval_name"]  = ""
        dict_eval["best_score"] = None
        dict_eval["best_result_list"] = []
    def _callback(env):
        if not dict_eval: _init(env)
        count = -1
        for data_name, eval_name, result, is_higher_better in env.evaluation_result_list:
            if data_name == "valid_0":
                count += 1
                if (isinstance(name, int) and count == name) or (isinstance(name, str) and eval_name == name):
                    if dict_eval["best_score"] is None:
                        dict_eval["best_score"] = result
                        dict_eval["eval_name"]  = eval_name
                        dict_eval["best_iter"]  = env.iteration
                        dict_eval["best_result_list"] = env.evaluation_result_list
                    else:
                        boolwk = result > dict_eval["best_score"] if is_higher_better else dict_eval["best_score"] > result
                        if boolwk:
                            dict_eval["best_score"] = result
                            dict_eval["eval_name"]  = eval_name
                            dict_eval["best_iter"]  = env.iteration
                            dict_eval["best_result_list"] = env.evaluation_result_list
                    break
        if isinstance(stopping_rounds, int) and ((env.iteration - dict_eval["best_iter"]) >= stopping_rounds):
            LOGGER.info(f'early stopping. iteration: {dict_eval["best_iter"]}, score: {dict_eval["best_score"]}')
            raise EarlyStopException(dict_eval["best_iter"], dict_eval["best_result_list"])
    _callback.order = 200
    return _callback


class _KkLogEvaluationCallback(_LogEvaluationCallback):
    def __call__(self, env: CallbackEnv) -> None:
        if self.period > 0 and env.evaluation_result_list and (env.iteration + 1) % self.period == 0:
            result = '\t'.join([_format_eval_result(x, self.show_stdv) for x in env.evaluation_result_list])
            LOGGER.info('[%d]\t%s' % (env.iteration, result))


def log_evaluation(period: int = 1, show_stdv: bool = True) -> _KkLogEvaluationCallback:
    return _KkLogEvaluationCallback(period=period, show_stdv=show_stdv)

