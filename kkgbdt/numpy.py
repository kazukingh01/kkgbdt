import numpy as np


__all__ = [
    "softmax",
    "sigmoid",
    "rmse",
    "mae",
    "sort_group_id",
    "evaluate_ndcg",
]


def softmax(x):
    """
    The code without njit is faster than with njit
    """
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_exp_x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rmse(x: np.ndarray, t: np.ndarray):
    return (x - t) ** 2 / 2.

def mae(x: np.ndarray, t: np.ndarray):
    return np.abs(x - t)

def sort_group_id(group: np.ndarray):
    ndf_idx = np.argsort(group)
    ndf     = group[ndf_idx]
    ndf_ids, ndf_cnt = np.unique(ndf, return_counts=True)
    ndf_test = np.concatenate([np.where(~(ndf[:-1] == ndf[1:]))[0], [ndf.shape[0] - 1]])
    assert np.all(ndf_ids == ndf[ndf_test])
    return ndf_idx, ndf_cnt

def evaluate_ndcg(ranks_pred: np.ndarray, ranks_answer: np.ndarray, points_answer: np.ndarray=None, k: int=None):
    """
    ranks_pred, ranks_answer:
        interger, 1 ~. 1 menas top rank
    idx | rank | point  ||  idx | pred  ||  point sort | rank idx | pred point
     0      1      6         0      1            6          0            6
     1      2      5         1      2            5          1            5
     2      6      1         2      3            4          3            1
     3      3      4         3      6            3          5            3
     4      5      2         4      5            2          4            2
     5      4      3         5      4            1          2            4
    >>> import numpy as np
    >>> ranks_pred    = np.array([[1, 2, 3, 6, 5, 4], [1, 3, 4, 5, 6, 2]])
    >>> ranks_answer  = np.array([[1, 2, 6, 3, 5, 4], [1, 2, 4, 5, 6,]])
    >>> points_answer = np.argsort(np.argsort(-ranks_answer, axis=-1), axis=-1) + 1
    >>> idx_answer    = np.argsort(-points_answer, axis=-1)
    >>> evaluate_ndcg(ranks_pred, ranks_answer)
    >>> evaluate_ndcg(ranks_pred, ranks_answer, k=3)
    """
    assert isinstance(ranks_pred,   np.ndarray)
    assert isinstance(ranks_answer, np.ndarray)
    assert len(ranks_pred.shape) in [1, 2]
    assert ranks_pred.shape == ranks_answer.shape
    assert (ranks_pred   < 1).sum() == 0
    assert (ranks_answer < 1).sum() == 0
    if points_answer is not None:
        assert isinstance(points_answer, np.ndarray)
        assert ranks_pred.shape == points_answer.shape
    else:
        points_answer = np.argsort(np.argsort(-ranks_answer, axis=-1), axis=-1) + 1
    assert k is None or (isinstance(k, int) and k > 0)
    if k is None:
        if len(ranks_pred.shape) == 1:
            k = ranks_pred.shape[0]
        else:
            k = ranks_pred.shape[-1]
    ndf    = np.log2(np.arange(k) + 1)
    p_pred = np.take_along_axis(points_answer, np.argsort(ranks_pred, axis=-1), axis=-1)
    if len(ranks_pred.shape) == 1:
        p_answer = np.sort(points_answer, axis=-1)[::-1]
    else:
        p_answer = np.sort(points_answer, axis=-1)[:, ::-1]
    if len(ranks_pred.shape) == 1:
        dcg_i = p_answer[0] + (p_answer[1:k] / ndf[1:k]).sum()
        dcg   = p_pred[  0] + (p_pred[  1:k] / ndf[1:k]).sum()
        return dcg / dcg_i
    else:
        ndf   = np.tile(ndf,      (ranks_pred.shape[0], 1))
        dcg_i = p_answer[:, 0] + (p_answer[:, 1:k] / ndf[:, 1:k]).sum(axis=-1)
        dcg   = p_pred[  :, 0] + (p_pred[  :, 1:k] / ndf[:, 1:k]).sum(axis=-1)
        return dcg / dcg_i
