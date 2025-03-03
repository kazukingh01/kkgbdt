import numpy as np
try: import polars as pl
except ImportError: pl = None


__all__ = [
    "softmax",
    "sigmoid",
    "rmse",
    "mae",
    "sort_group_id",
    "evaluate_ndcg",
    "log_loss",
]


def softmax(x):
    """
    The code without njit is faster than with njit
    """
    shift_x   = x - np.max(x, axis=-1, keepdims=True) # To avoid overflow encountered in exp
    exp_x     = np.exp(shift_x)
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

def evaluate_ndcg(
    ranks_pred: np.ndarray[float | int], ranks_answer: np.ndarray[int], points: dict[int, int]=None,
    k: int=None, idx_groups: list[int | str] | np.ndarray=None, is_point_to_rank: bool=False
):
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
    assert (ranks_answer < 1).sum() == 0
    assert isinstance(is_point_to_rank, bool)
    if is_point_to_rank == False:
        assert (ranks_pred < 1).sum() == 0
    assert k is None or (isinstance(k, int) and k > 0)
    if idx_groups is not None:
        assert isinstance(idx_groups, (list, np.ndarray))
        if isinstance(idx_groups, list):
            idx_groups = np.array(idx_groups)
        assert len(idx_groups.shape) == 1
        assert idx_groups.shape == ranks_pred.shape
        assert k is not None
        df = pl.DataFrame([
            pl.Series(ranks_pred  ).alias("pred"),
            pl.Series(ranks_answer).alias("answer"),
            pl.Series(idx_groups  ).alias("group"),
        ])
        df = df.group_by("group").agg(pl.all())
        ranks_pred   = df.select([pl.col("pred"  ).list.get(i, null_on_oob=True).alias(f"{i}") for i in range(k)]).to_numpy()
        ranks_answer = df.select([pl.col("answer").list.get(i, null_on_oob=True).alias(f"{i}") for i in range(k)]).to_numpy()
    if is_point_to_rank:
        ndf_nan    = np.isnan(ranks_pred)
        ranks_pred = (np.argsort(np.argsort(-ranks_pred, axis=-1)) + 1).astype(float)
        ranks_pred[ndf_nan] = float("nan")
    if points is not None:
        assert isinstance(points_answer, dict)
        points_answer = np.vectorize(lambda x: points.get(x) if x in points else float("nan"), otypes=[float])(ranks_answer)
    else:
        points_answer = (np.argsort(np.argsort(-ranks_answer, axis=-1), axis=-1) + 1).astype(float)
        points_answer[np.isnan(ranks_answer)] = float("nan")
    if k is None:
        if len(ranks_pred.shape) == 1:
            k = ranks_pred.shape[0]
        else:
            k = ranks_pred.shape[-1]
    ndf      = np.log2(np.arange(k) + 1)
    ndf[0]   = 1
    p_pred   = np.take_along_axis(points_answer, np.argsort(ranks_pred, axis=-1), axis=-1)
    p_answer = -np.sort(-points_answer, axis=-1)
    if len(ranks_pred.shape) > 1:
        ndf   = np.tile(ndf, (ranks_pred.shape[0], 1))
    dcg_i = np.nansum(p_answer[..., :k] / ndf[..., :k], axis=-1)
    dcg   = np.nansum(p_pred[  ..., :k] / ndf[..., :k], axis=-1)
    return dcg / dcg_i

def log_loss(y: np.ndarray, x: np.ndarray):
    assert isinstance(y, np.ndarray) and len(y.shape) == 1
    assert isinstance(x, np.ndarray) and len(x.shape) == 2
    assert y.dtype in [int, np.int8, np.int16, np.int32, np.int64]
    ndf = x[np.arange(x.shape[0]), y]
    return (-1 * np.log(ndf)).mean()
