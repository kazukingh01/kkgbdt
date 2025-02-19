import time
from xgboost.callback import TrainingCallback
from lightgbm.callback import EarlyStopException, _format_eval_result, _LogEvaluationCallback, CallbackEnv
from kklogger import set_logger
from .com import check_type


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
    def __init__(
        self, train_stopping_val: int | float=None, train_stopping_rounds: int=None,
        train_stopping_is_over: bool=True, train_stopping_time: int | float=None
    ):
        super().__init__()
        assert train_stopping_val  is None or check_type(train_stopping_val,  [int, float])
        assert train_stopping_time is None or check_type(train_stopping_time, [int, float])
        assert isinstance(train_stopping_rounds, int) and train_stopping_rounds > 0
        assert isinstance(train_stopping_is_over, bool)
        self.train_stopping_val     = train_stopping_val
        self.train_stopping_rounds  = train_stopping_rounds
        self.train_stopping_is_over = train_stopping_is_over
        self.train_stopping_time    = train_stopping_time
        self.dict_train             = {"time": float("inf")}
    def after_iteration(self, model, epoch: int, evals_log: TrainingCallback.EvalsLog) -> bool:
        if (self.train_stopping_time is not None) and ((time.time() - self.dict_train["time"]) > self.train_stopping_time) and epoch < self.train_stopping_rounds:
            LOGGER.info("Stopping training because time is over.")
            return True
        if self.train_stopping_val is not None and epoch >= self.train_stopping_rounds:
            dictwk = evals_log["train"]
            valwk  = dictwk[list(dictwk.keys())[0]]
            boolwk = (valwk[-1] > self.train_stopping_val) if self.train_stopping_is_over else (valwk[-1] < self.train_stopping_val)
            if boolwk:
                LOGGER.info(f"Stopping training because loss value is {'over' if self.train_stopping_is_over else 'less'} than {self.train_stopping_val}.")
                return True
        self.dict_train["time"] = time.time()
        return False


def callback_stop_training(dict_train: dict, train_stopping_val: int | float, train_stopping_rounds: int, train_stopping_is_over: bool, train_stopping_time: int | float):
    """
    If training loss does not reach the threshold, it will be terminated first.
    """
    def _init(env):
        dict_train["time"] = float("inf")
    def _callback(env):
        if not dict_train: _init(env)
        _, _, result, _ = env.evaluation_result_list[0]
        boolwk = False
        if check_type(train_stopping_time, [int, float]) and ((time.time() - dict_train["time"]) > train_stopping_time) and env.iteration < train_stopping_rounds:
            # In case training step time is too slow within "stopping_rounds" steps
            boolwk = True
        if check_type(train_stopping_val, [float, int]) and isinstance(train_stopping_rounds, int) and env.iteration >= train_stopping_rounds:
            # In case training result is too bad over "stopping_rounds" steps
            if train_stopping_is_over and result > train_stopping_val:
                boolwk = True
            elif train_stopping_is_over == False and result < train_stopping_val:
                boolwk = True
        if boolwk:
            LOGGER.info(f'stop training. iteration: {env.iteration}, score: {result}')
            raise EarlyStopException(env.iteration, env.evaluation_result_list)
        dict_train["time"] = time.time()
    _callback.order = 150
    return _callback


def callback_best_iter(dict_eval: dict, stopping_rounds: int, name: str | int):
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

