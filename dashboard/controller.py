# dashboard/controller.py

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.multi_timeframe_models import train_models_per_timeframe, predict_models_per_timeframe
from utils.signal_voting import vote_signals
from pipeline.backtest import simple_strategy_backtest
from pipeline.evaluator import (
    plot_equity_curve,
    plot_predictions,
    plot_confusion,
    plot_rolling_sharpe
)
from utils.plot_utils import create_plot_dir
from pipeline.data_loader import normalize_features
import os


def main():
    st.set_page_config(page_title="Timeframe Model Trainer", layout="wide")
    st.title("⏱️ Multi-Timeframe Model Trainer & Evaluator")

    data_dir = Path("data")
    csv_files = sorted([f for f in data_dir.glob("gold_*.csv")])

    timeframes = [f.stem.replace("gold_", "") for f in csv_files]
    selected_timeframes = st.multiselect("Select timeframes to include:", timeframes)

    if not selected_timeframes:
        st.warning("Please select at least one timeframe to continue.")
        return

    selected_tasks = st.multiselect("Select training modes:", ["Train per timeframe", "Train combined"],
                                    default=["Train per timeframe", "Train combined"])
    models_to_use = st.multiselect("Choose models to train:", ["XGBoost", "LSTM", "CNN-LSTM", "TCN", "Transformer"],
                                   default=["XGBoost"])

    run_button = st.button("🚀 Run Training")
    if run_button:
        dfs = []
        for tf in selected_timeframes:
            path = data_dir / f"gold_{tf}.csv"
            if path.exists():
                df = pd.read_csv(path, parse_dates=['Date'], index_col=None)
                df.columns = [f"{col}_{tf}" if col != "Date" else "Date" for col in df.columns]
                dfs.append(df)

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
        st.subheader("📊 Strategy Evaluation")
        st.pyplot(plot_equity_curve(backtest_df, plot_dir=plot_dir, model_name="dashboard"))
        st.pyplot(plot_predictions(df, plot_dir=plot_dir, model_name="dashboard"))
        st.pyplot(plot_confusion(df, plot_dir=plot_dir, model_name="dashboard"))
        st.pyplot(plot_rolling_sharpe(backtest_df, plot_dir=plot_dir, model_name="dashboard"))


if __name__ == '__main__':
    main()
