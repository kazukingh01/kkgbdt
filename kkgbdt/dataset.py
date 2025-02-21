from lightgbm import Dataset
import numpy as np
# local package
from kklogger import set_logger
from .functions import sort_group_id

__all__ = [
    "DatasetLGB",
]


LOGGER = set_logger(__name__)


class DatasetLGB(Dataset):
    """
    Params::
        group: None | np.ndarray | list[np.ndarray]
            Group structure of the dataset.
            If np.ndarray, like ["aa", "aa", "aa", "bb", "bb", "cc", ...], the dataset will be sorted by group and "group" will be changed to official input style.
    Usage::
        >>> dataset = KkLgbDataset(x_train)
        >>> dataset.set_culstom_label(y_train)
    """
    def __init__(self, data: np.ndarray, *args, label: np.ndarray=None, group: None | np.ndarray=None, **kwargs):
        assert isinstance(data, np.ndarray) and len(data.shape) == 2
        assert label is None or isinstance(label, np.ndarray)
        if group is not None:
            assert isinstance(group, np.ndarray) and len(group.shape) == 1
            assert group.dtype in [int, np.int32, np.int64, object]
            if group.shape[0] == data.shape[0]:
                LOGGER.warning("This group structure is complicated so dataset will be sorted by group.")
                ndf_idx, group = sort_group_id(group)
                data = data[ndf_idx]
                if label is not None:
                    label = label[ndf_idx]
            else:
                assert group.dtype in [int, np.int32, np.int64]
                assert group.sum() == data.shape[0]
        super().__init__(data, *args, label=label, group=group, **kwargs)
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


