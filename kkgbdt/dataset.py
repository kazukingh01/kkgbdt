from xgboost.core import DMatrix
import numpy as np


__all__ = [
    "Dataset"
]


class Dataset(DMatrix):
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
