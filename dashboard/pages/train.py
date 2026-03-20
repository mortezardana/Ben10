# pages/train.py
"""Legacy training page — uses the original multi-timeframe approach.
Updated with safe imports and optional pipeline integration.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import traceback

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Safe imports with fallbacks
_IMPORTS_OK = True
_IMPORT_ERRORS = []

try:
    from models.multi_timeframe_models import train_models_per_timeframe, predict_models_per_timeframe
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERRORS.append(f"multi_timeframe_models: {e}")

try:
    from utils.signal_voting import vote_signals
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERRORS.append(f"signal_voting: {e}")

try:
    from pipeline.backtest import simple_strategy_backtest
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERRORS.append(f"backtest: {e}")

try:
    from pipeline.evaluator import (
        plot_equity_curve,
        plot_predictions,
        plot_confusion,
        plot_rolling_sharpe
    )
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERRORS.append(f"evaluator: {e}")

try:
    from utils.plot_utils import create_plot_dir
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERRORS.append(f"plot_utils: {e}")

try:
    from pipeline.data_loader import normalize_features
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERRORS.append(f"data_loader: {e}")


def run():
    st.title("Train & Evaluate Models (Legacy)")

    if not _IMPORTS_OK:
        st.warning(
            "Some imports failed. The legacy training page may not work fully. "
            "Consider using the **Pipeline Training** page instead."
        )
        with st.expander("Import errors"):
            for err in _IMPORT_ERRORS:
                st.code(err)

    st.markdown(
        "This page uses the original multi-timeframe training approach. "
        "For the new standardized pipeline interface, use **Pipeline Training** from the sidebar."
    )

    data_dir = Path("data")
    if not data_dir.exists():
        data_dir = Path("Data")
    csv_files = sorted([f for f in data_dir.glob("gold_*.csv")]) if data_dir.exists() else []

    if not csv_files:
        st.warning("No gold_*.csv files found in the data directory.")
        return

    timeframes = [f.stem.replace("gold_", "") for f in csv_files]
    selected_timeframes = st.multiselect("Select timeframes to include:", timeframes)

    if not selected_timeframes:
        st.warning("Please select at least one timeframe to continue.")
        return

    selected_tasks = st.multiselect("Select training modes:", ["Train per timeframe", "Train combined"],
                                    default=["Train per timeframe", "Train combined"])
    models_to_use = st.multiselect("Choose models to train:", ["XGBoost", "LSTM", "CNN-LSTM", "TCN", "Transformer"],
                                   default=["XGBoost"])

    run_button = st.button("Run Training")
    if run_button:
        if not _IMPORTS_OK:
            st.error("Cannot run training due to import errors listed above.")
            return

        try:
            dfs = []
            for tf in selected_timeframes:
                path = data_dir / f"gold_{tf}.csv"
                if path.exists():
                    df = pd.read_csv(path, parse_dates=['Date'], index_col=None)
                    df.columns = [f"{col}_{tf}" if col != "Date" else "Date" for col in df.columns]
                    dfs.append(df)

            if not dfs:
                st.error("No data files found for selected timeframes.")
                return

            # Merge all selected
            df_merged = dfs[0]
            for df in dfs[1:]:
                df_merged = df_merged.merge(df, on="Date", how="inner")

            df = df_merged.copy()

            # Add binary target column
            close_col = [col for col in df.columns if col.lower().endswith("close") or "close_" in col.lower()][0]
            df['target'] = (df[close_col].shift(-1) > df[close_col]).astype(int)
            df.dropna(inplace=True)

            df, _ = normalize_features(df, exclude_cols=['Date', 'target'])
            plot_dir = create_plot_dir()

            if "Train per timeframe" in selected_tasks:
                model_objs = train_models_per_timeframe(df, selected_timeframes, target_col='target')
                df = predict_models_per_timeframe(df, selected_timeframes)
                if len(selected_timeframes) >= 2:
                    fast, slow = selected_timeframes[:2]
                    df = vote_signals(df, col_fast=f"pred_{fast}", col_slow=f"pred_{slow}", mode="confirm")
                    df['prediction'] = df['voted_signal']
                    st.success("Voting complete between first 2 timeframes")
                else:
                    df['prediction'] = df[f"pred_{selected_timeframes[0]}"]

            if "Train combined" in selected_tasks:
                from models.baseline_ml import train_baseline_ml
                model, metrics = train_baseline_ml(df, target_col='target')
                from models.predictor import predict
                df['prediction'] = predict(model, df)
                st.success("Combined model trained and predicted")

            # Backtest and evaluate
            backtest_df = simple_strategy_backtest(df, price_col=close_col)
            st.dataframe(backtest_df.tail())
            st.subheader("Strategy Evaluation")
            st.pyplot(plot_equity_curve(backtest_df, plot_dir=plot_dir, model_name="dashboard"))
            st.pyplot(plot_predictions(df, plot_dir=plot_dir, model_name="dashboard"))
            st.pyplot(plot_confusion(df, plot_dir=plot_dir, model_name="dashboard"))
            st.pyplot(plot_rolling_sharpe(backtest_df, plot_dir=plot_dir, model_name="dashboard"))

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.code(traceback.format_exc())
