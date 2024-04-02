import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from compare_parameters import PARAMS_FIX


if __name__ == "__main__":
    df = pd.read_pickle("df_eval.pickle")
    columns_fix = list(PARAMS_FIX.keys())
    df["eval"] = df["eval"].replace(float("inf"), float("nan")).replace(float("-inf"), float("nan"))
    df = df.loc[:, ~df.columns.isin(columns_fix)]

    # use_quantized_grad = False
    dfwk    = df.loc[df["use_quantized_grad"] == False].copy()
    df_plot = dfwk.groupby(["max_bin", "num_leaves", "max_depth", "mode"])[["time", "eval"]].mean()
    df_plot[["time_std", "eval_std"]] = dfwk.groupby(["max_bin", "num_leaves", "max_depth", "mode"])[["time", "eval"]].std()
    fig1, ax1 = plt.subplots(2, 1, figsize=(17, 10))
    ax1     = ax1.flatten() if hasattr(ax1, "flatten") else np.array([ax1])
    for name in ["xgb", "lgb"]:
        dfwkwk = df_plot.loc[(slice(None), slice(None),slice(None), name)]
        ax1[0].errorbar(
            np.arange(dfwkwk.shape[0]), dfwkwk["eval"], yerr=dfwkwk["eval_std"],
            fmt="o", color={"xgb": "red", "lgb": "blue"}[name], capsize=2, alpha=0.5, label=name
        )
        ax1[1].errorbar(
            np.arange(dfwkwk.shape[0]), dfwkwk["time"], yerr=dfwkwk["time_std"],
            fmt="o", color={"xgb": "red", "lgb": "blue"}[name], capsize=2, alpha=0.5, label=name
        )
    ax1[0].set_ylabel("logloss")
    ax1[1].set_ylabel("time")
    ax1[0].legend()
    ax1[1].legend()
    ax1[0].set_xticks(np.arange(dfwkwk.shape[0]), [f"{x}".replace(" ", "").replace(",", "\n") for x in dfwkwk.index])
    ax1[1].set_xticks(np.arange(dfwkwk.shape[0]), [f"{x}".replace(" ", "").replace(",", "\n") for x in dfwkwk.index])
    ax1[0].grid(linestyle='dotted', c='gray', linewidth=1)
    ax1[1].grid(linestyle='dotted', c='gray', linewidth=1)

    # use_quantized_grad = True
    dfwk    = df.loc[df["use_quantized_grad"] == True].copy()
    df_plot = dfwk.groupby(["max_bin", "num_leaves", "max_depth", "mode", "num_grad_quant_bins"])[["time", "eval"]].mean()
    df_plot[["time_std", "eval_std"]] = dfwk.groupby(["max_bin", "num_leaves", "max_depth", "mode", "num_grad_quant_bins"])[["time", "eval"]].std()
    fig2, ax2 = plt.subplots(2, 1, figsize=(17, 10))
    ax2     = ax2.flatten() if hasattr(ax2, "flatten") else np.array([ax2])
    for num_grad_quant_bins in dfwk["num_grad_quant_bins"].unique():
        dfwkwk = df_plot.loc[(slice(None), slice(None) ,slice(None), slice(None), num_grad_quant_bins)]
        ax2[0].errorbar(
            np.arange(dfwkwk.shape[0]), dfwkwk["eval"], yerr=dfwkwk["eval_std"],
            fmt="o", color={4: "red", 8: "blue", 16: "green"}[num_grad_quant_bins], capsize=2, alpha=0.5, label=f"bins: {num_grad_quant_bins}"
        )
        ax2[1].errorbar(
            np.arange(dfwkwk.shape[0]), dfwkwk["time"], yerr=dfwkwk["time_std"],
            fmt="o", color={4: "red", 8: "blue", 16: "green"}[num_grad_quant_bins], capsize=2, alpha=0.5, label=f"bins: {num_grad_quant_bins}"
        )
    ax2[0].set_ylabel("logloss")
    ax2[1].set_ylabel("time")
    ax2[0].legend()
    ax2[1].legend()
    ax2[0].set_xticks(np.arange(dfwkwk.shape[0]), [f"{x[:-1]}".replace(" ", "").replace(",", "\n") for x in dfwkwk.index])
    ax2[1].set_xticks(np.arange(dfwkwk.shape[0]), [f"{x[:-1]}".replace(" ", "").replace(",", "\n") for x in dfwkwk.index])
    ax2[0].grid(linestyle='dotted', c='gray', linewidth=1)
    ax2[1].grid(linestyle='dotted', c='gray', linewidth=1)

    # lim
    min_y0 = min(ax1[0].get_ylim()[0], ax2[0].get_ylim()[0])
    max_y0 = max(ax1[0].get_ylim()[1], ax2[0].get_ylim()[1])
    min_y1 = min(ax1[1].get_ylim()[0], ax2[1].get_ylim()[0])
    max_y1 = max(ax1[1].get_ylim()[1], ax2[1].get_ylim()[1])
    ax1[0].set_ylim(min_y0, max_y0)
    ax2[0].set_ylim(min_y0, max_y0)
    ax1[1].set_ylim(min_y1, max_y1)
    ax2[1].set_ylim(min_y1, max_y1)
    fig1.savefig("xgb_vs_lgb.png")
    fig2.savefig("lgb_quantized.png")