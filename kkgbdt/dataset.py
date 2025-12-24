import os
from lightgbm import Dataset
from xgboost import DMatrix, QuantileDMatrix
from catboost import Pool
import numpy as np
# local package
from kklogger import set_logger
from .functions import sort_group_id

__all__ = [
    "DatasetLGB",
    "DatasetCB",
]


LOGGER = set_logger(__name__)


def check_and_sort_group_id(
    data: np.ndarray, group: np.ndarray, label: np.ndarray=None, weight: np.ndarray=None,
    check_mode: str = "xgb"
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    assert isinstance(data, np.ndarray)
    assert isinstance(group, np.ndarray) and len(group.shape) == 1
    assert group.dtype in [int, np.int32, np.int64, object]
    LOGGER.info(f"[ {check_mode} ] group before: {group}")
    if label is not None:
        assert isinstance(label, np.ndarray)
        assert label.shape[0] == data.shape[0]
    if weight is not None:
        assert isinstance(weight, np.ndarray) and len(weight.shape) == 1
        assert weight.shape[0] == data.shape[0]
    assert check_mode in ["xgb", "lgb", "cat"]
    if check_mode in ["xgb", "lgb"]:
        if group.shape[0] == data.shape[0]:
            ndf_idx, group = sort_group_id(group)
            data = data[ndf_idx]
            if label  is not None: label  = label[ ndf_idx]
            if weight is not None: weight = weight[ndf_idx]
        else:
            assert group.dtype in [int, np.int32, np.int64]
            assert group.sum() == data.shape[0]
    else:
        if group.shape[0] == data.shape[0]:
            pass # No check
        else:
            assert group.dtype in [int, np.int32, np.int64]
            assert group.sum() == data.shape[0]
            group = [i for i, c in enumerate(group) for _ in range(c)]
    LOGGER.info(f"[ {check_mode} ] group after: {group}")
    return data, group, label, weight


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
    def __init__(
        self, data: np.ndarray | str, *args, label: np.ndarray=None, weight: np.ndarray=None,
        group: None | np.ndarray=None, **kwargs
    ):
        assert isinstance(data, (np.ndarray, str))
        if isinstance(data, np.ndarray): assert len(data.shape) == 2
        assert label  is None or isinstance(label,  np.ndarray)
        assert weight is None or isinstance(weight, np.ndarray)
        if group is not None:
            data, group, label, weight = check_and_sort_group_id(data, group, label=label, weight=weight, check_mode="lgb")
        super().__init__(data, *args, label=label, weight=weight, group=group, **kwargs)
        if isinstance(data, np.ndarray): 
            if label is not None and len(label.shape) == 2:
                LOGGER.info(f"set custom dataset. x_train shape: {data.shape}, {data.dtype} | y_train shape: {label.shape}, {label.dtype}")
                self.set_custom_label(label)
            else:
                LOGGER.info(f"set normal dataset. x_train shape: {data.shape}, {data.dtype} | y_train shape: {label.shape if label is not None else None}, {label.dtype if label is not None else None}")
        else:
            assert os.path.isfile(data), f"file not found: {data}"
            LOGGER.info(f"load binary dataset: {data}")
    def set_custom_label(self, label: np.ndarray):
        self.custom_label = label.copy()
        self.label        = np.arange(label.shape[0]).astype(int)
    def get_custom_label(self, indexes: np.ndarray) -> np.ndarray:
        return self.custom_label[indexes.astype(int)]


class DatasetCB(Pool):
    def __init__(
        self, data: np.ndarray, label: np.ndarray=None, weight: np.ndarray=None, group: np.ndarray=None, 
        cat_features: list[int]=None, **kwargs
    ):
        assert isinstance(data, np.ndarray) and len(data.shape) == 2
        assert label  is None or isinstance(label,  np.ndarray)
        assert weight is None or isinstance(weight, np.ndarray)
        if group is not None:
            data, group, label, weight = check_and_sort_group_id(data, group, label=label, weight=weight, check_mode="cat")
        if group is not None:
            super().__init__(data=data, label=label, weight=weight, group_id=group, cat_features=cat_features, **kwargs)
        else:
            super().__init__(data=data, label=label, weight=weight, cat_features=cat_features, **kwargs)
        LOGGER.info(f"set catboost dataset. x_train shape: {data.shape}, {data.dtype} | y_train shape: {label.shape if label is not None else None}, {label.dtype if label is not None else None}")

def create_xgb_dataset_with_group(
    data: np.ndarray, label: np.ndarray=None, weight: np.ndarray=None,
    group: None | np.ndarray=None, use_quantile: bool=False, max_bin: int=None, **kwargs
) -> DMatrix | QuantileDMatrix:
    assert isinstance(data, np.ndarray) and len(data.shape) == 2
    assert label  is None or isinstance(label,  np.ndarray)
    assert weight is None or isinstance(weight, np.ndarray)
    assert isinstance(use_quantile, bool)
    if group is not None:
        data, group, label, weight = check_and_sort_group_id(data, group, label=label, weight=weight, check_mode="xgb")
    if use_quantile:
        assert isinstance(max_bin, int) and max_bin > 0
        dataset = QuantileDMatrix(data, label=label, weight=weight, max_bin=max_bin, **kwargs)
    else:
        dataset = DMatrix(data, label=label, weight=weight, **kwargs)
    if group is not None:
        dataset.set_group(group)
    LOGGER.info(f"set normal dataset. x_train shape: {data.shape}, {data.dtype} | y_train shape: {label.shape if label is not None else None}, {label.dtype if label is not None else None}")
    return dataset
