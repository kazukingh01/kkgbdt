from lightgbm import Dataset
import numpy as np
from kklogger import set_logger


__all__ = [
    "DatasetLGB",
]


LOGGER = set_logger(__name__)


class DatasetLGB(Dataset):
    """
    Usage::
        >>> dataset = KkLgbDataset(x_train)
        >>> dataset.set_culstom_label(y_train)
    """
    def __init__(self, data, *args, label: np.ndarray=None, **kwargs):
        super().__init__(data, *args, label=label, **kwargs)
        if label is not None and len(label.shape) == 2:
            LOGGER.info(f"set custom dataset. x_train shape: {data.shape}, {data.dtype} | y_train shape: {label.shape}, {label.dtype}")
            self.set_custom_label(label)
        else:
            LOGGER.info(f"set normal dataset. x_train shape: {data.shape}, {data.dtype} | y_train shape: {label.shape if label is not None else None}, {label.dtype if label is not None else None}")
    def set_custom_label(self, label: np.ndarray):
        self.custom_label = label.copy()
        self.label        = np.arange(label.shape[0]).astype(int)
    def get_custom_label(self, indexes: np.ndarray) -> np.ndarray:
        return self.custom_label[indexes.astype(int)]
