import numpy as np
import pandas as pd
from lightgbm.engine import Booster
from kklogger import set_logger


__all__ = [
    "KkTracer",
]


LOGGER = set_logger(__name__)


class KkTracer(Booster):
    def create_tree_dataframe(self):
        LOGGER.info("START")
        dftree = self.trees_to_dataframe()
        dftree["leaf_index"] = (dftree["node_index"].str.split("-").str[-1].str[1:]).astype(int).astype(float)
        dftree.loc[~dftree["threshold"].isna(), "leaf_index"] = float("nan")
        dict_index = {y:x for x, y in dftree["node_index"].to_dict().items()}
        dftree["left_child" ] = dftree["left_child" ].map(dict_index)
        dftree["right_child"] = dftree["right_child"].map(dict_index)
        dfwk = dftree[["node_index"]].copy()
        dfwk.columns  = ["parent_index"]
        dfwk["index"] = dfwk.index.copy()
        dftree = pd.merge(dftree, dfwk, how="left", on=["parent_index"])
        sewk   = dftree["index"].isna()
        dftree.loc[sewk, "index"] = dftree.index[sewk]
        dftree["index"]       = dftree["index"].astype(int)
        dftree["colname"]     = dftree["split_feature"].str.replace("^Column_", "", regex=True).fillna(-1).astype(int)
        self.tracer_dftree    = dftree.copy()
        dfwk                  = (self.tracer_dftree.groupby("tree_index")["leaf_index"].max() + 1).reset_index()
        dfwk["tree_index"]   += 1
        dfwk                  = pd.concat([dfwk, pd.DataFrame([[0, 0]], columns=["tree_index", "leaf_index"])], ignore_index=True, sort=False)
        dfwk                  = dfwk.sort_values("tree_index").reset_index(drop=True)
        dfwk["leaf_index"]    = dfwk["leaf_index"].cumsum()
        dfwk.columns          = ["tree_index", "leaf_index_add"]
        self.tracer_dftree    = pd.merge(self.tracer_dftree, dfwk, how="left", on="tree_index")
        self.tracer_index_add = dfwk["leaf_index_add"].iloc[:-1].astype(int).values.copy().reshape(1, -1)
        dfwk                  = self.tracer_dftree["leaf_index"] + self.tracer_dftree["leaf_index_add"]
        self.tracer_leaf      = dfwk.dropna().astype(int).reset_index().sort_values(0)["index"].values.copy()
        self.tracer_dfmat     = self.create_path_matrix(self.tracer_leaf.reshape(1, -1), aggregate_func="mean", n_deep=2)
        logger.info("END")

    def create_path_matrix(self, tracer_leaf: np.ndarray, aggregate_func: str="sum", n_deep: int=1):
        logger.info("START")
        assert isinstance(aggregate_func, str) and aggregate_func in ["mean", "sum"]
        assert isinstance(n_deep, int) and n_deep > 0
        _, _, ndf_thre, ndf_gain, ndf_col = self.interpret_path(tracer_leaf)
        ndf_col_thre = ndf_col.astype(str).astype(object) + "_" + ndf_thre.astype(str).astype(object)
        list_ndf_col_thre, list_ndf_gain = [], []
        for i_deep in range(1, n_deep + 1):
            list_ndf_col_thre.append(ndf_col_thre[:, :-i_deep, :] + "__" + ndf_col_thre[:, i_deep:, :])
            list_ndf_gain.append(ndf_gain[:, :-i_deep, :] + ndf_gain[:, i_deep:, :])
        ndf_col_thre = np.concatenate(list_ndf_col_thre, axis=-2)
        ndf_gain     = np.concatenate(list_ndf_gain,     axis=-2)
        ndf_col_thre[np.isnan(ndf_gain)] = None
        dfmat         = pd.DataFrame(ndf_col_thre.reshape(-1), columns=["colname"])
        dfmat["gain"] = ndf_gain.reshape(-1)
        dfmat         = getattr(dfmat.groupby("colname")["gain"], aggregate_func)().reset_index()
        for i_col in [1, 2]:
            dfmat[f"colname{i_col}"]      = dfmat["colname"].str.split("__").str[i_col - 1]
            dfmat[f"colname{i_col}_main"] = dfmat[f"colname{i_col}"].str.split("_").str[0]
            dfmat[f"colname{i_col}_thre"] = dfmat[f"colname{i_col}"].str.split("_").str[-1]
        dfwk = pd.DataFrame(pd.concat([dfmat["colname1"], dfmat["colname2"]], ignore_index=True, sort=False).reset_index().groupby(0).size().index.values, columns=["colname"]).reset_index()
        dfwk["colname_main"] = dfwk["colname"].str.split("_").str[0 ].astype(int)
        dfwk["colname_thre"] = dfwk["colname"].str.split("_").str[-1].astype(float)
        dfwk["colname_thre"] = dfwk.groupby("colname_main")["colname_thre"].rank().astype(int)
        for i_col in [1, 2]:
            dfwkwk = dfwk[["index", "colname"]].copy()
            dfwkwk.columns = [f"index{i_col}", f"colname{i_col}"]
            dfmat = pd.merge(dfmat, dfwkwk, how="left", on=f"colname{i_col}")
        ndf = np.zeros((dfwk.shape[0], dfwk.shape[0]))
        ndf[dfmat["index1"].values, dfmat["index2"].values] = dfmat["gain"].values.copy()
        tmp = dfwk[["colname_main", "colname_thre"]].apply(lambda x: tuple(x), axis=1)
        df  = pd.DataFrame(ndf, index=tmp.tolist(), columns=tmp.tolist())
        logger.info("END")
        return df

    def predict_leaf_index(self, test_x, *args, **kwargs):
        logger.info("START")
        leaf_index = self.predict(test_x, *args, pred_leaf=True, **kwargs)
        if len(leaf_index.shape) == 1: leaf_index = leaf_index.reshape(-1, 1)
        leaf_index    = leaf_index + self.tracer_index_add
        leaf_df_index = self.tracer_leaf[leaf_index]
        logger.info("END")
        return leaf_df_index
    
    def interpret_path(self, leaf_df_index: np.ndarray):
        logger.info("START")
        assert isinstance(leaf_df_index, np.ndarray) and len(leaf_df_index.shape) == 2
        leaf_values   = self.tracer_dftree["value"].values[leaf_df_index]
        ndf_path      = leaf_df_index.reshape(-1, 1, leaf_df_index.shape[-1])
        ndf_bool      = np.zeros_like(ndf_path).astype(bool)
        tmp           = self.tracer_dftree["index"].values
        while True:
            ndfwk    = tmp[ndf_path[:, -1:, :]]
            ndfwkwk  = (ndf_path[:, -1:, :] == ndfwk)
            if np.all(ndfwkwk): break
            ndf_path = np.concatenate([ndf_path, ndfwk],   axis=1)
            ndf_bool = np.concatenate([ndf_bool, ndfwkwk], axis=1)
        ndf_path[ndf_bool] = self.tracer_dftree.shape[0] + 1
        ndf_path = np.sort(ndf_path, axis=1)
        ndf_path[ndf_path == self.tracer_dftree.shape[0] + 1] = -1
        logger.info("END")
        return (
            leaf_values,
            ndf_path,
            self.tracer_dftree["threshold" ].values[ndf_path],
            self.tracer_dftree["split_gain"].values[ndf_path],
            self.tracer_dftree["colname"   ].values[ndf_path]
        )
