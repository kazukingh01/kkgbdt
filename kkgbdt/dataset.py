from xgboost.core import DMatrix
import numpy as np


__all__ = [
    "Dataset"
]


class Dataset(DMatrix):
    """
    Usage::
        >>> dataset = Dataset(x_train)
        >>> dataset.set_culstom_label(y_train)
    """
    def set_culstom_label(self, label: np.ndarray):
        self.label        = np.arange(label.shape[0], dtype=int)
        self.custom_label = label
    def get_culstom_label(self, indexes: np.ndarray):
        return self.custom_label[indexes]
