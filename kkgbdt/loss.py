import numpy as np
import xgboost as xgb
from functools import partial
from scipy.misc import derivative
from scipy.stats import norm
from kkgbdt.dataset import DatasetLGB
from kkgbdt.util.numpy import sigmoid, softmax
from kkgbdt.util.com import check_type_list, check_type
from kklogger import set_logger


LOGGER = set_logger(__name__)


__all__ = [
    "Loss",
    "LGBCustomObjective",
    "LGBCustomEval",
    "calc_grad_hess",
    "BinaryCrossEntropyLoss",
    "CrossEntropyLoss",
    "CrossEntropyLossArgmax",
    "CategoricalCrossEntropyLoss",
    "FocalLoss",
    "MSELoss",
    "MAELoss",
    "HuberLoss",
    "Accuracy",
    "LogitMarginL1Loss",
    "MultiTaskLoss",
    "MultiTaskEvalLoss",
    "CrossEntropyNDCGLoss",
    "DensitySoftmax",
]


def _return(x): return x
def _return_1(x): return 1.0
def _return_binary(x): return np.stack([1 - x, x]).T
def _reshape(x): return x.reshape(-1)
def _sum_reshape(x): return x.sum(axis=1).reshape(-1, 1)


class Loss:
    def __init__(
            self, name: str, n_classes: int=1, reduction: str="mean", is_higher_better: bool=False
        ):
        assert isinstance(n_classes, int) and n_classes >= 0
        assert isinstance(reduction, str) and reduction in ["mean", "sum"]
        assert isinstance(is_higher_better, bool)
        self.name         = name
        self.n_classes    = n_classes
        self.conv_shape   = _return
        self.is_check     = True
        self.is_prob      = False
        self.reduction    = np.mean if reduction == "mean" else np.sum
        self.is_higher_better = is_higher_better
    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"
    def convert(self, x: np.ndarray, t: np.ndarray):
        if self.is_check:
            self.check(x, t)
            if self.n_classes == 1: self.conv_shape = _reshape
            self.is_check = False
        x = self.conv_shape(x)
        return x, t
    def check(self, x: np.ndarray, t: np.ndarray):
        assert isinstance(x, np.ndarray)
        assert isinstance(t, np.ndarray)
        LOGGER.debug(f"\ninput: {x}\ninput shape: {x.shape}\ninput dtype: {x.dtype}\nlabel: {t}\nlabel shape: {t.shape}\nlabel dtype: {t.dtype}")
        assert x.shape[0] == t.shape[0]
        if self.n_classes > 1: assert x.shape[1] == self.n_classes
    def __call__(self, x: np.ndarray, t: np.ndarray):
        loss = self.loss(x, t)
        return self.reduction(loss)
    def loss(self, x: np.ndarray, t: np.ndarray):
        raise NotImplementedError
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        raise NotImplementedError


class LGBCustomObjective:
    def __init__(self, func_loss: Loss, mode: str="xgb"):
        assert isinstance(func_loss, Loss)
        assert isinstance(mode, str) and mode in ["xgb", "lgb"]
        self.func_loss = func_loss
        self.mode      = mode
        if   self.mode == "xgb":
            self.conv_input  = self.convert_xgb_input
            self.comp_weight = self.compute_xgb_weight
            self.conv_output = self.convert_xgb_output
        elif self.mode == "lgb":
            self.conv_input  = self.convert_lgb_input
            self.comp_weight = self.compute_lgb_weight
            self.conv_output = self.convert_lgb_output
    def __call__(self, y_pred: np.ndarray, data):
        y_pred, y_true = self.conv_input(y_pred, data)
        grad, hess = self.func_loss.gradhess(y_pred, y_true)
        grad, hess = self.comp_weight(grad, hess, data)
        grad, hess = self.conv_output(grad, hess)
        return grad, hess
    @classmethod
    def convert_xgb_input(cls, y_pred: np.ndarray, data: xgb.DMatrix):
        y_true = data.get_label()
        if y_pred.shape[0] != y_true.shape[0]:
            # multi_strategy: "multi_output_tree" mode
            # see: https://xgboost.readthedocs.io/en/stable/python/examples/multioutput_regression.html#sphx-glr-python-examples-multioutput-regression-py
            y_true = data.get_label().reshape(y_pred.shape)
        return y_pred, y_true
    @classmethod
    def convert_lgb_input(cls, y_pred: np.ndarray, data: DatasetLGB):
        """
        Params::
            y_pred:
                Predicted value. In the case of multi class, the length is n_sample * n_class
                Value is ... array([0 data 0 label prediction, ..., N data 0 label prediction, 0 data 1 label prediction, ..., ])
            data:
                Dataset
        """
        if hasattr(data, "custom_label"):
            y_true = data.get_custom_label(data.label)
        else:
            y_true = data.label
        if y_pred.shape[0] != y_true.shape[0]:
            # multi class case
            y_pred = y_pred.reshape(-1 , y_true.shape[0]).T
        return y_pred, y_true
    @classmethod
    def compute_xgb_weight(cls, grad: np.ndarray, hess: np.ndarray, data: xgb.DMatrix):
        weight = data.get_weight()
        if weight is None or len(weight) == 0: return grad, hess
        if len(grad.shape) == 2:
            weight = weight.reshape(-1, 1)
        grad   = grad * weight
        hess   = hess * weight
        return grad, hess
    @classmethod
    def compute_lgb_weight(cls, grad: np.ndarray, hess: np.ndarray, data: DatasetLGB):
        weight = data.get_weight()
        if weight is None or len(weight) == 0: return grad, hess
        if len(grad.shape) == 2:
            weight = weight.reshape(-1, 1)
        grad   = grad * weight
        hess   = hess * weight
        return grad, hess
    @classmethod
    def convert_xgb_output(cls, grad: np.ndarray, hess: np.ndarray):
        return grad, hess
    @classmethod
    def convert_lgb_output(cls, grad: np.ndarray, hess: np.ndarray):
        return grad.T.reshape(-1), hess.T.reshape(-1)


class LGBCustomEval(LGBCustomObjective):
    def __init__(self, func_loss: Loss, mode: str="xgb", name: str=None, is_higher_better: bool=None):
        if name             is None: name             = func_loss.name
        if is_higher_better is None: is_higher_better = func_loss.is_higher_better
        assert isinstance(name, str)
        assert isinstance(is_higher_better, bool)
        super().__init__(func_loss, mode=mode)
        self.name = name
        self.is_higher_better = is_higher_better
        if   mode == "xgb":
            self.conv_metric = self.convert_xgb_metric
        elif mode == "lgb":
            self.conv_metric = self.convert_lgb_metric
    def __call__(self, y_pred: np.ndarray, data):
        y_pred, y_true = self.conv_input(y_pred, data)
        value = self.func_loss(y_pred, y_true)
        return self.conv_metric(value)
    def convert_xgb_metric(self, value):
        return self.name, value
    def convert_lgb_metric(self, value):
        return self.name, value, self.is_higher_better


def calc_grad_hess(x: np.ndarray, t: np.ndarray, loss_func=None, dx=1e-6, **kwargs):
    grad = derivative(lambda _x: loss_func(_x, t, **kwargs), x, n=1, dx=dx)
    hess = derivative(lambda _x: loss_func(_x, t, **kwargs), x, n=2, dx=dx)
    return grad, hess


class BinaryCrossEntropyLoss(Loss):
    def __init__(self, dx: float=0):
        """
        Params::
            dx: Log loss is undefined for p=0 or p=1, so probabilities are
                clipped to `max(eps, min(1 - eps, p))`. The default will depend on the
                data type of `y_pred` and is set to `np.finfo(y_pred.dtype).eps`.
        """
        assert check_type(dx, [float, int]) and dx >= 0.0 and dx < 1.0
        super().__init__("bce", n_classes=1, is_higher_better=False)
        self.dx      = dx
        self.is_prob = True
        self.is_clip = (dx > 0)
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        assert np.isin(np.unique(t), [0,1]).sum() == 2
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x = sigmoid(x)
        if self.is_clip: x = np.clip(x, self.dx, 1.0 - self.dx)
        return -1 * (t * np.log(x) + (1 - t) * np.log(1 - x))
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x = sigmoid(x)
        if self.is_clip: x = np.clip(x, self.dx, 1.0 - self.dx)
        grad = x - t
        hess = (1 - x) * x
        return grad, hess


class CrossEntropyLoss(Loss):
    def __init__(self, n_classes: int, dx: float=0):
        assert isinstance(n_classes, int) and n_classes > 1
        assert check_type(dx, [float, int]) and dx >= 0.0 and dx < 1.0
        super().__init__("ce", n_classes=n_classes, is_higher_better=False)
        self.dx         = dx
        self.is_clip    = (dx > 0)
        self.is_prob    = True
        self.conv_t_sum = _return_1
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        if self.name == "ce":
            assert len(x.shape) == 2
            assert len(t.shape) == 2
            assert x.shape[1] == t.shape[1] == self.n_classes
            assert (t > 1).sum() == (t < 0).sum() == 0
            if ((t.sum(axis=1) / 1e-6).round(0).astype(np.int32) == int(round(1 / 1e-6, 0))).sum() != t.shape[0]:
                # If the sum of "t" is not equal to 1 (In other words, if "t" is not a probability)
                self.conv_t_sum = _sum_reshape
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x = softmax(x)
        if self.is_clip: x = np.clip(x, self.dx, 1.0 - self.dx)
        return (-1 * t * np.log(x)).sum(axis=1)
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        """
        see: https://qiita.com/klis/items/4ad3032d02ff815e09e6
        see: https://www.ics.uci.edu/~pjsadows/notes.pdf
        """
        x, t = self.convert(x, t)
        x = softmax(x)
        if self.is_clip: x = np.clip(x, self.dx, 1.0 - self.dx)
        t_sum  = self.conv_t_sum(t)
        t_sum2 = t_sum * x
        grad   = t_sum2 - t
        hess   = t_sum2 * (1 - x)
        return grad, hess


class CrossEntropyLossArgmax(Loss):
    def __init__(self, n_classes: int, dx: float=0):
        assert isinstance(n_classes, int) and n_classes > 1
        assert check_type(dx, [float, int]) and dx >= 0.0 and dx < 1.0
        super().__init__("cemax", n_classes=n_classes, is_higher_better=False)
        self.dx      = dx
        self.is_clip = (dx > 0)
        self.is_prob = True
        self.conv_t  = _return_1
        self.indexes = None
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        if self.name == "cemax":
            assert len(x.shape) == 2
            assert len(t.shape) in [1, 2]
            if len(t.shape) == 2:
                assert (t > 1).sum() == (t < 0).sum() == 0
                assert x.shape[1] == t.shape[1] == self.n_classes
                self.conv_t = partial(np.argmax, axis=1)
            else:
                assert np.isin(np.unique(t), np.arange(self.n_classes)).sum() == self.n_classes
                assert x.shape[1] == self.n_classes
                self.conv_t = partial(np.astype, dtype=np.int32)
            self.indexes = np.arange(t.shape[0] * 10, dtype=int) # x 10, in case shape is different at each iteration
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        t = self.conv_t(t)
        x = softmax(x)
        if self.is_clip: x = np.clip(x, self.dx, 1.0 - self.dx)
        x = x[self.indexes[:t.shape[0]], t]
        return (-1 * np.log(x))
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        raise Exception(f"class: {self.__class__.__name__} has not gradient and hessian.")


class CategoricalCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, n_classes: int, dx: float=0, smoothing: int | float=0):
        assert check_type(smoothing, [int, float]) and 0.0 <= smoothing < 1.0
        assert check_type(dx, [float, int]) and dx >= 0.0 and dx < 1.0
        super().__init__(n_classes, dx=dx)
        self.smoothing = smoothing
        self.is_smooth = self.smoothing > 0.0
        self.is_prob   = True
        self.name      = f"cce({format(smoothing, '.0e')})" if self.is_smooth else "cce"
        self.ndfid     = None
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        assert len(x.shape) == 2
        assert len(t.shape) == 1
        assert np.isin(np.unique(t), np.arange(self.n_classes)).sum() == self.n_classes
        assert (t - t.astype(int)).sum() == 0 # There is no fraction
        if self.is_smooth:
            self.ndfid = np.identity(x.shape[1], dtype=bool)
        else:
            self.ndfid = np.identity(x.shape[1], dtype=np.float32)
    def convert(self, x: np.ndarray, t: np.ndarray):
        x, t = super().convert(x, t)
        t = self.ndfid[t.astype(int)]
        if self.is_smooth:
            ndf    = np.full(x.shape, self.smoothing / (self.n_classes - 1))
            ndf[t] = 1 - self.smoothing
            t      = ndf.astype(np.float32)
        return x, t


class FocalLoss(Loss):
    def __init__(self, n_classes: int, gamma: float=1.0, dx: float=0):
        assert isinstance(n_classes, int) and n_classes > 1
        assert isinstance(gamma, float) and gamma >= 0
        assert check_type(dx, [float, int]) and dx >= 0.0 and dx < 1.0
        super().__init__(f"fl({gamma})", n_classes=n_classes, is_higher_better=False)
        self.gamma   = gamma
        self.dx      = dx
        self.is_clip = (dx > 0)
        self.is_prob = True
        self.ndfid   = None
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        assert len(x.shape) == 2
        assert len(t.shape) == 1
        assert np.isin(np.unique(t), np.arange(self.n_classes)).sum() == self.n_classes
        assert (t - t.astype(int)).sum() == 0 # There is no fraction
        self.ndfid = np.identity(x.shape[1], dtype=bool)
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x = softmax(x)
        if self.is_clip: x = np.clip(x, self.dx, 1.0 - self.dx)
        t = self.ndfid[t.astype(int)]
        return -1 * ((1 - x[t]) ** self.gamma) * np.log(x[t])
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        """
        see: https://hackmd.io/Kd14LHwETwOXLbzh_adpQQ
        """
        x, t = self.convert(x, t)
        x    = softmax(x)
        if self.is_clip: x = np.clip(x, self.dx, 1.0 - self.dx)
        t    = self.ndfid[t.astype(int)]
        yk   = x[t].reshape(-1, 1)
        yk_log  = np.log(yk)
        yk_p0   = self.gamma * yk
        yk_p1   = yk_p0 * yk_log
        yk_p2   = 1 - yk
        yk_p3   = yk_p2 ** self.gamma
        yk_p4   = 1 - (yk + yk_p1)
        grad    = x * (yk_p2 ** (self.gamma - 1)) * yk_p4
        grad[t] = (-yk_p3 * yk_p4).reshape(-1)
        hess    = x * (yk_p2 ** (self.gamma - 2)) * (
            (1 - x - yk + yk_p0 * x ) * yk_p4 + x * yk * yk_p2 * (self.gamma * yk_log + self.gamma + 1)
        )
        hess[t] = (yk * yk_p3 * (self.gamma * yk_log * (1 - yk - yk_p0) + 1 - yk - 2 * yk_p0 + 2 * self.gamma)).reshape(-1)
        return grad, hess


class MSELoss(Loss):
    def __init__(self, n_classes: int=1):
        assert isinstance(n_classes, int) and n_classes > 0
        super().__init__("mse", n_classes=n_classes, is_higher_better=False)
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        return 0.5 * (x - t) ** 2
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        grad = x - t
        hess = np.ones(x.shape, dtype=np.float32)
        return grad, hess


class MAELoss(Loss):
    def __init__(self, n_classes: int=1):
        assert isinstance(n_classes, int) and n_classes > 0
        super().__init__("mae", n_classes=n_classes, is_higher_better=False)
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        return np.abs(x - t)
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        grad = np.ones(x.shape, dtype=np.float32)
        grad[x < t] = -1
        hess = np.zeros(x.shape, dtype=np.float32)
        return grad, hess


class HuberLoss(Loss):
    def __init__(self, n_classes: int=1, beta: float=1.0):
        assert isinstance(n_classes, int) and n_classes > 0
        super().__init__("huber", n_classes=n_classes, is_higher_better=False)
        self.beta = beta
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        diff   = x - t
        boolwk = ((-self.beta <= diff) & (diff <= self.beta))
        loss   = self.beta * np.abs(diff) - 0.5 * (self.beta ** 2) 
        loss[boolwk] = 0.5 * (diff[boolwk] ** 2)
        return loss
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        diff   = x - t
        boolwk = ((-self.beta <= diff) & (diff <= self.beta))
        grad = np.ones(t.shape, dtype=np.float32) * self.beta
        grad[diff < 0] = -1.0 * self.beta
        grad[boolwk] = (x - t)[boolwk]
        hess = np.zeros(t.shape, dtype=np.float32)
        hess[boolwk] = 1.0
        return grad, hess


class Accuracy(Loss):
    def __init__(self, top_k: int=1):
        assert isinstance(top_k, int) and top_k >= 1
        self.top_k = top_k
        super().__init__(f"acc_top{self.top_k}", n_classes=0, is_higher_better=True)
        self.conv_shape_x = _return
        self.conv_shape_t = _return
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        assert len(x.shape) in [1, 2]
        if len(x.shape) == 1:
            self.conv_shape_x = _return_binary
        assert len(t.shape) in [1, 2]
        if len(t.shape) == 2:
            assert (t > 1).sum() == (t < 0).sum() == 0
            self.conv_shape_t = partial(np.argmax, axis=1)
        else:
            assert np.isin(np.unique(t), np.arange(self.n_classes)).sum() == self.n_classes
            assert (t - t.astype(int)).sum() == 0 # There is no fraction
    def convert(self, x: np.ndarray, t: np.ndarray):
        x, t = super().convert(x, t)
        x = self.conv_shape_x(x)
        t = self.conv_shape_t(t).astype(int)
        return x, t
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x = np.argsort(x, axis=1)[:, ::-1]
        x = x[:, :self.top_k]
        return (x == t.reshape(-1, 1)).sum(axis=1)
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        raise Exception(f"class: {self.__class__.__name__} has not gradient and hessian.")


class LogitMarginL1Loss(Loss):
    def __init__(
        self, n_classes: int, alpha: float=0.1, margin: float=1.0, dx: int | float=0
    ):
        """
        https://arxiv.org/pdf/2111.15430.pdf
        https://github.com/by-liu/MbLS/blob/167c0267b7a0ae29b255d44af0589a88af4d2410/calibrate/losses/logit_margin_l1.py
        """
        assert isinstance(n_classes, int) and n_classes > 1
        assert isinstance(alpha,  float) and alpha  > 0
        assert isinstance(margin, float) and margin > 0
        assert check_type(dx, [float, int]) and dx >= 0.0 and dx < 1.0
        self.alpha  = alpha
        self.margin = margin
        super().__init__(f"ce_margin_{self.alpha}_{self.margin}", n_classes=n_classes, is_higher_better=False)
        self.dx         = dx
        self.is_clip    = (dx > 0)
        self.is_prob    = True
        self.conv_t_sum = _return_1
        self.ndfid      = None
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        assert len(x.shape) == 2
        assert len(t.shape) in [1, 2]
        if len(t.shape) == 2:
            assert x.shape[1] == t.shape[1] == self.n_classes
            if ((t.sum(axis=1) / 1e-6).round(0).astype(np.int32) == int(round(1 / 1e-6, 0))).sum() != t.shape[0]:
                # If the sum of "t" is not equal to 1 (In other words, if "t" is not a probability)
                self.conv_t_sum = _sum_reshape
        else:
            assert np.isin(np.unique(t), np.arange(self.n_classes)).sum() == self.n_classes
            assert (t - t.astype(int)).sum() == 0 # There is no fraction
            self.ndfid = np.identity(x.shape[1], dtype=np.float32)
    def convert(self, x: np.ndarray, t: np.ndarray):
        x, t = super().convert(x, t)
        if self.ndfid is not None:
            t = self.ndfid[t.astype(int)]
        return x, t
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x_margin = np.clip(np.max(x, axis=1).reshape(-1, 1) - x - self.margin, 0.0, None)
        x = softmax(x)
        if self.is_clip: x = np.clip(x, self.dx, 1.0 - self.dx)
        loss_ce     = (-1 * t * np.log(x)).sum(axis=1)
        loss_margin = self.alpha * np.sum(x_margin, axis=1)
        return loss_ce + loss_margin
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x_margin = np.max(x, axis=1).reshape(-1, 1) - x - self.margin
        x_margin = (x_margin > 0).astype(x.dtype)
        x = softmax(x)
        if self.is_clip: x = np.clip(x, self.dx, 1.0 - self.dx)
        t_sum  = self.conv_t_sum(t)
        t_sum2 = t_sum * x
        grad   = t_sum2 - t - (self.alpha * x_margin)
        hess   = t_sum2 * (1 - x)
        return grad, hess


class MultiTaskLoss(Loss):
    def __init__(self, losses: list[Loss], weight: list[float]=None):
        assert check_type_list(losses, Loss)
        if weight is None: weight = [1.0] * len(losses)
        assert check_type_list(weight, float)
        indexes_loss = np.cumsum([x.n_classes for x in losses])
        super().__init__(f"multi_loss", n_classes=int(indexes_loss[-1]), is_higher_better=False)
        self.indexes_loss = indexes_loss
        self.losses       = losses
        self.weight       = weight
    def check(self, x: np.ndarray, t: np.ndarray):
        assert x.shape == t.shape
        super().check(x, t)
    def convert(self, x: np.ndarray, t: np.ndarray):
        if self.is_check:
            self.check(x, t)
            self.is_check = False
        x = np.hsplit(x, self.indexes_loss[:-1])
        t = np.hsplit(t, self.indexes_loss[:-1])
        return x, t
    def loss(self, x: np.ndarray, t: np.ndarray):
        loss = np.zeros(x.shape[0], dtype=float)
        x, t = self.convert(x, t)
        for _x, _t, loss_func, _w in zip(x, t, self.losses, self.weight):
            loss += _w * loss_func.loss(_x, _t)
        return loss
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        list_grad, list_hess = [], []
        for _x, _t, loss_func, _w in zip(x, t, self.losses, self.weight):
            grad, hess = loss_func.gradhess(_x, _t)
            grad, hess = _w * grad, _w * hess
            list_grad.append(grad)
            list_hess.append(hess)
        return np.concatenate(list_grad, axis=1), np.concatenate(list_hess, axis=1)


class MultiTaskEvalLoss(Loss):
    def __init__(self, loss_func: Loss, indexes_loss: list[int]):
        assert isinstance(loss_func, Loss)
        assert check_type_list(indexes_loss, int)
        super().__init__(loss_func.name, n_classes=loss_func.n_classes, is_higher_better=loss_func.is_higher_better)
        self.loss_func    = loss_func
        self.indexes_loss = indexes_loss
    def check(self, x: np.ndarray, t: np.ndarray):
        assert x.shape == t.shape
    def convert(self, x: np.ndarray, t: np.ndarray):
        if self.is_check:
            self.check(x, t)
            self.is_check = False
        x, t = x[:, self.indexes_loss], t[:, self.indexes_loss]
        return x, t
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        return self.loss_func.loss(x, t)
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        raise Exception(f"class: {self.__class__.__name__} has not gradient and hessian.")


class CrossEntropyNDCGLoss(Loss):
    def __init__(self, n_classes: int, eta: list[float]=None):
        """
        https://arxiv.org/pdf/1911.09798.pdf
        """
        super().__init__("xendcg", n_classes=n_classes, is_higher_better=False)
        self.is_prob = True
        if eta is not None:
            assert check_type_list(eta, [float, int])
            self.eta = np.array(eta).astype(float)
        else:
            self.eta = 0.0
    def check(self, x: np.ndarray, t: np.ndarray):
        assert x.shape == t.shape
        if isinstance(self.eta, np.ndarray):
            assert x.shape[-1] == self.eta.shape[-1]
        super().check(x, t)
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        ro   = softmax(x)
        phi  = np.power(2.0, t) - self.eta
        phi  = phi / phi.sum(axis=1).reshape(-1, 1)
        return (-1 * phi * np.log(ro)).sum(axis=1)
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        ro   = softmax(x)
        phi  = np.power(2.0, t) - self.eta
        phi  = phi / phi.sum(axis=1).reshape(-1, 1)
        grad = -phi + ro
        hess = ro * (1 - ro)
        return grad, hess
    @classmethod
    def DCG(cls, x: np.ndarray, t: np.ndarray):
        x_rank = (np.argsort(np.argsort(-x, axis=-1), axis=1) + 1)
        return ((np.power(2, t) - 1) / np.log2(1 + x_rank)).sum(axis=1)
    @classmethod
    def NDCG(cls, x: np.ndarray, t: np.ndarray):
        return __class__.DCG(x, t) / __class__.DCG(t, t)


class DensitySoftmax(Loss):
    def __init__(self, n_input: int, n_classes: int, learning_rate: float=1e-2, dx: float=1e-10, maxsmpl: int=10000, epoch: int=100):
        """
        https://arxiv.org/abs/2302.06495
        https://hackmd.io/4VuiVOA4SfisvylasPdiAA
        I couldn't find out formular of Probability Density Function
            p(z_t, a) * z_t^T theta_i (It doesn't match dimentions)
        """
        assert isinstance(n_input,   int) and n_input   > 2
        assert isinstance(n_classes, int) and n_classes > 2
        assert isinstance(learning_rate, float) and learning_rate > 0.
        assert isinstance(maxsmpl, int) and maxsmpl > 0
        assert isinstance(epoch, int) and epoch > 0
        super().__init__("dence", n_classes=n_classes, is_higher_better=False)
        self.lr      = learning_rate
        self.dx      = dx
        self.is_prob = True
        self.weight  = np.random.rand(n_input, n_classes)
        self.bias    = np.random.rand(1,       n_classes)
        self.maxsmpl = maxsmpl
        self.epoch   = epoch
        self.mu      = float("nan")
        self.sigma   = float("nan")
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        assert len(x.shape) == 2
        assert len(t.shape) == 1
        assert np.isin(np.unique(t), np.arange(self.n_classes)).sum() == self.n_classes
    def convert(self, x: np.ndarray, t: np.ndarray):
        x, t = super().convert(x, t)
        t = np.identity(x.shape[1])[t].astype(np.float32)
        return x, t
    def classifier(self, x: np.ndarray):
        F = np.einsum("ab,bc->ac", x, self.weight)
        F = F + self.bias
        F = softmax(F)
        F = np.clip(F, self.dx, 1 - self.dx)
        return F
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        F    = self.classifier(x)
        return (-1 * t * np.log(F)).sum(axis=1)
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        """
        command to compare to pytorch::
            # gradient for x
            x = torch.rand(1, 4, requires_grad=True)
            t = torch.randint(0, 3, (1,)).to(torch.long)
            t = torch.eye(3)[t]
            weight = torch.nn.Linear(4, 3)
            def _func(x, t=None): return torch.nn.CrossEntropyLoss(reduction="sum")(weight(x), t)

            func = partial(_func, t=t)
            loss = func(x)
            grad = torch.autograd.grad(loss, x, create_graph=True)[0]
            print(grad)
            t_sum = t.sum(dim=-1).reshape(-1, 1)
            F = weight(x)
            F = torch.softmax(F, dim=-1)
            _grad = torch.einsum("ab,bc->ac", t_sum * F - t, weight.weight)
            print(_grad)
            # hessian for x 
            hess = torch.autograd.functional.hessian(func, x)
            print(hess)
            _hess = t_sum * (torch.einsum("ab,bc->ac", F, weight.weight ** 2) - ((torch.einsum("ab,bc->ac", F, weight.weight)) ** 2))
            print(_hess)
            # gradient for weight
            gradw = torch.autograd.grad(loss, weight.weight, create_graph=True)[0]
            print(gradw)
            _gradw = torch.einsum("ab,ac->abc", x, F * t_sum - t).T
            print(_gradw)
            # gradient for bias
            gradb = torch.autograd.grad(loss, weight.bias, create_graph=True)[0]
            print(gradb)
            _gradb = F * t_sum - t
            print(_gradb)
        """
        x, t = self.convert(x, t)
        F    = self.classifier(x)
        tsum = t.sum(axis=-1).reshape(-1, 1)
        # grad, hess for GBDT
        grad = np.einsum("ab,bc->ac", F * tsum - t, self.weight.T)
        hess = tsum * (np.einsum("ab,bc->ac", F, self.weight.T ** 2) - (np.einsum("ab,bc->ac", F, self.weight.T) ** 2))
        # grad for weight
        smpl  = np.random.permutation(x.shape[0])[:min(x.shape[0], self.maxsmpl)]
        _x    = x[smpl]
        _t    = t[smpl]
        _F    = F[smpl]
        _tsum = tsum[smpl]
        _grad = np.einsum("ab,bc->bac", _x.T, (_F * _tsum - _t))
        _grad = _grad.mean(axis=0)
        self.weight = self.weight - (self.lr * _grad)
        _grad = (_F * _tsum - _t)
        _grad = _grad.mean(axis=0).reshape(1, -1)
        self.bias = self.bias - (self.lr * _grad)
        return grad, hess
    def inference(self, x: np.ndarray, *args, **kwargs):
        _x = np.einsum("ab,bc->ac", x, self.weight) + self.bias
        p  = norm.pdf(x, loc=self.mu, scale=self.sigma)
        p  = p * _x
        f  = np.exp(p)/np.sum(np.exp(p), axis=1, keepdims=True)
        return f
    def extra_processing(self, x: np.ndarray, t: np.ndarray, epoch: int=None):
        if epoch is None: epoch = self.epoch
        assert isinstance(epoch, int)
        # calc mu & sigma & norm.pdf ( this is first )
        x, t = self.convert(x, t)
        F    = self.classifier(x)
        tsum = t.sum(axis=-1).reshape(-1, 1)
        self.mu    = np.mean(x)
        self.sigma = np.sqrt(np.var(x))
        p          = norm.pdf(x, loc=self.mu, scale=self.sigma) # MLE
        p          = p / p.max() # scale
        for i_epoch in range(epoch):
            smpl  = np.random.permutation(x.shape[0])[:min(x.shape[0], self.maxsmpl)]
            _x    = x[smpl]
            _t    = t[smpl]
            _F    = F[smpl]
            _p    = p[smpl]
            _tsum = tsum[smpl]
            if self.sigma > 1e-5:
                # on first training, all output is 0 so _p goes to nan. I want to avoid it.
                _E    = _F * _p
                _grad = np.einsum("ab,bc->bac", _x.T, _p * (_E * _tsum - _t))
                _grad = _grad.mean(axis=0)
                self.weight = self.weight - (self.lr * _grad)
                _grad = _p * (_E * _tsum - _t)
                _grad = _grad.mean(axis=0).reshape(1, -1)
                self.bias = self.bias - (self.lr * _grad)
            F    = self.classifier(x)
            loss = (-1 * t * np.log(F)).sum(axis=1)
            loss = self.reduction(loss)
            print(f"epoch: {i_epoch}, loss: {loss}")
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        assert len(x.shape) == 2
        assert len(t.shape) == 1
        assert np.isin(np.unique(t), np.arange(self.n_classes)).sum() == self.n_classes
        assert (t - t.astype(int)).sum() == 0 # There is no fraction
        if self.is_smooth:
            self.ndfid = np.identity(x.shape[1], dtype=bool)
        else:
            self.ndfid = np.identity(x.shape[1], dtype=np.float32)
    def convert(self, x: np.ndarray, t: np.ndarray):
        x, t = super().convert(x, t)
        t = self.ndfid[t.astype(int)]
        if self.is_smooth:
            ndf    = np.full(x.shape, self.smoothing / (self.n_classes - 1))
            ndf[t] = 1 - self.smoothing
            t      = ndf.astype(np.float32)
        return x, t
