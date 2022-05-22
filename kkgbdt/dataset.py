from xgboost.core import DMatrix
from lightgbm import Dataset
import numpy as np
from kkgbdt.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "DatasetXGB",
    "DatasetLGB",
]


class DatasetXGB(DMatrix):
    """
    Usage::
        >>> dataset = Dataset(x_train)
        >>> dataset.set_custom_label(y_train)
    """
    def __init__(self, *args, label: np.ndarray=None, **kwargs):
        if label is not None and len(label.shape) == 2:
            label = self.set_custom_label(label)
        super().__init__(*args, label=label, **kwargs)
    def set_custom_label(self, label: np.ndarray):
        assert isinstance(label, np.ndarray) and len(label.shape) == 2
        self.custom_label = label.copy()
        self.n_data       = label.shape[0]
        return np.arange(label.shape[0], dtype=float) / label.shape[0]
    def get_custom_label(self, indexes: np.ndarray):
        return self.custom_label[(indexes * self.n_data).astype(int)]


class DatasetLGB(Dataset):
    """
    Usage::
        >>> dataset = KkLgbDataset(x_train)
        >>> dataset.set_culstom_label(y_train)
    """
    def __init__(self, data, *args, label: np.ndarray=None, **kwargs):
        super().__init__(data, *args, label=label, **kwargs)
        if label is not None and len(label.shape) == 2:
            logger.info(f"set custom dataset. x_train shape: {data.shape}, y_train shape: {label.shape}")
            self.set_custom_label(label)
        else:
            logger.info(f"set normal dataset. x_train shape: {data.shape}, y_train shape: {label.shape if label is not None else None}")
    def set_custom_label(self, label: np.ndarray):
        self.label        = np.arange(label.shape[0]).astype(int)
        self.custom_label = label
    def get_custom_label(self, indexes: np.ndarray) -> np.ndarray:
        return self.custom_label[indexes.astype(int)]
