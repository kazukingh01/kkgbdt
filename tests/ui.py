import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from kklogger import set_logger


LOGGER = set_logger(__name__)
MODE   = {
    "lgb": sqlite3.connect("params_lgb.db"),
    "xgb": sqlite3.connect("params_xgb.db"),
    "cat": sqlite3.connect("params_cat.db"),
}


def get_studies(conn: sqlite3.Connection) -> list[str]:
    assert isinstance(conn, sqlite3.Connection)
    try:
        df = pd.read_sql_query("SELECT study_name FROM studies ORDER BY study_id;", conn)
    except pd.errors.DatabaseError as e:
        LOGGER.error(f"Failed to get studies: {e}")
        return []
    return df["study_name"].tolist()

def get_trial_data(conn: sqlite3.Connection, study_name: str, prefix: str = "params_lgb_") -> pd.DataFrame:
    assert isinstance(conn, sqlite3.Connection)
    assert isinstance(study_name, str) and study_name
    assert isinstance(prefix, str)
    try:
        id_study  = pd.read_sql_query(f"SELECT study_id FROM studies WHERE study_name = '{prefix + study_name}' ORDER BY study_id;", conn)["study_id"].iloc[-1]
    except IndexError:
        return pd.DataFrame()
    df_trials = pd.read_sql_query(
        f"""
        SELECT trial_id, number, state, datetime_start, datetime_complete
        FROM trials 
        WHERE study_id = {id_study}
        ORDER BY number;
        """.strip(), conn
    )
    if df_trials.empty:
        return pd.DataFrame()
    ids_trial = df_trials["trial_id"].astype(str).tolist()
    # get values
    df_values = pd.read_sql_query(f"SELECT trial_id, objective as pname, value                 FROM trial_values          WHERE trial_id IN ({",".join(ids_trial)});", conn)
    df_params = pd.read_sql_query(f"SELECT trial_id, param_name as pname, param_value as value FROM trial_params          WHERE trial_id IN ({",".join(ids_trial)});", conn)
    df_attrs  = pd.read_sql_query(f"SELECT trial_id, key as pname, value_json as value         FROM trial_user_attributes WHERE trial_id IN ({",".join(ids_trial)});", conn)
    df_values["pname"] = "objective"
    df_attrs["value"]  = df_attrs["value"].astype(float)
    # join
    df = pd.concat([df_values, df_params, df_attrs], axis=0, ignore_index=True)
    df = df.pivot_table(values="value", index="trial_id", columns="pname", aggfunc="mean").reset_index()
    if "time_iter" not in df.columns:
        df["time_iter"] = df["time_train"] / df["total_iteration"]
    return df.drop(columns=["trial_id"]).reset_index(drop=True)


if __name__ == "__main__":
    # meta
    st.set_page_config(
        page_title="Optuna Study Viewer",
        page_icon="📊",
        layout="wide"
    )    
    st.title("📊 Optuna Study Viewer")

    # sidebar
    with st.sidebar:
        selected_dataset = st.sidebar.selectbox(
            "Select Dataset",
            options=[str(f).replace("params_lgb_", "") for f in get_studies(MODE["lgb"])],
            index=0
        )
        st.sidebar.markdown("**Select Package**")
        selected_modes = [False] * len(MODE)
        for i, (mode, _) in enumerate(MODE.items()):
            selected_modes[i] = st.sidebar.checkbox(mode, value=True)
        selected_modes = [mode for mode, chk in zip(MODE.keys(), selected_modes) if chk]

    # get data
    list_df = []
    for mode, conn in MODE.items():
        dfwk = get_trial_data(conn, selected_dataset, prefix=f"params_{mode}_")
        dfwk["mode"] = mode
        list_df.append(dfwk)
    df = pd.concat(list_df, axis=0, ignore_index=True)

    # display dataframe
    with st.expander("📄 Data Preview", expanded=False):
        st.dataframe(df, use_container_width=True)

    # plot
    st.subheader("📈 2D Scatter Plot")
    df_filtered = df.loc[df["mode"].isin(selected_modes)]
    if df_filtered.empty:
        st.warning("Selected package has no data.")
    else:
        cols_numeric = df_filtered.select_dtypes(include=["number"]).columns.tolist()
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X axis", options=cols_numeric, index=0)
            x_log = st.checkbox("Log scale (X)", value=False)
        with col2:
            y_col = st.selectbox("Y axis", options=cols_numeric, index=1)
            y_log = st.checkbox("Log scale (Y)", value=False)
        with col3:
            z_col = st.selectbox("Value",  options=cols_numeric, index=None)
            z_log = st.checkbox("Log scale (Value)", value=False, disabled=(z_col is None))
        # scatter plot
        df_plot = df_filtered.copy()
        if z_log and z_col is not None:
            df_plot[f"{z_col}_log"] = np.log10(df_plot[z_col].clip(lower=1e-10))
        fig = px.scatter(
            df_plot,
            x=x_col,
            y=y_col,
            color=f"{z_col}_log" if z_log and z_col is not None else z_col,
            symbol="mode",
            symbol_map={"lgb": "star", "cat": "square-open-dot", "xgb": "circle-open"},
            color_continuous_scale="Viridis",
            hover_data=df_filtered.columns.tolist(),
            title=f"{x_col} vs {y_col} (Value: {z_col if z_log and z_col is not None else f'log10({z_col})'})"
        )
        fig.update_layout(
            height=600,
            coloraxis_colorbar=dict(title=z_col)
        )
        fig.update_traces(marker=dict(size=9))
        if x_log:
            fig.update_xaxes(type="log")
        if y_log:
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, width="stretch")
