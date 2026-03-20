# dashboard/controller.py
"""Standalone controller for multi-timeframe model training.
Updated to also support the new TradingPipeline interface.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import traceback

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Legacy imports (with fallbacks)
try:
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
    _LEGACY_IMPORTS_OK = True
except ImportError as e:
    _LEGACY_IMPORTS_OK = False
    _LEGACY_IMPORT_ERROR = str(e)

# New pipeline imports (with fallbacks)
try:
    from shared.interfaces import TradingPipeline, PipelineOutput
    from shared.data_loader import load_gold_data
    from shared.config import load_config
    from pipeline.backtest import BacktestEngine
    _PIPELINE_IMPORTS_OK = True
except ImportError as e:
    _PIPELINE_IMPORTS_OK = False
    _PIPELINE_IMPORT_ERROR = str(e)

import os


def _get_pipeline_registry():
    """Lazily build pipeline registry."""
    registry = {}
    pipeline_imports = [
        ("XGBoost", "pipelines.gradient_boosting.xgboost_pipeline", "XGBoostPipeline"),
        ("LightGBM", "pipelines.gradient_boosting.lightgbm_pipeline", "LightGBMPipeline"),
        ("CatBoost", "pipelines.gradient_boosting.catboost_pipeline", "CatBoostPipeline"),
        ("LSTM", "pipelines.deep_learning.lstm_pipeline", "LSTMPipeline"),
        ("GRU", "pipelines.deep_learning.gru_pipeline", "GRUPipeline"),
        ("CNN-LSTM", "pipelines.deep_learning.cnn_lstm_pipeline", "CNNLSTMPipeline"),
        ("TCN", "pipelines.deep_learning.tcn_pipeline", "TCNPipeline"),
        ("Transformer", "pipelines.deep_learning.transformer_pipeline", "TransformerPipeline"),
        ("TabNet", "pipelines.deep_learning.tabnet_pipeline", "TabNetPipeline"),
        ("CRNN", "pipelines.deep_learning.crnn_pipeline", "CRNNPipeline"),
        ("HMM Regime", "pipelines.regime.hmm_pipeline", "HMMRegimePipeline"),
    ]
    for display_name, module_path, class_name in pipeline_imports:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            registry[display_name] = getattr(mod, class_name)
        except (ImportError, AttributeError):
            pass
    return registry


def main():
    st.set_page_config(page_title="Timeframe Model Trainer", layout="wide")
    st.title("Multi-Timeframe Model Trainer & Evaluator")

    mode = st.sidebar.radio("Mode:", ["Legacy (Multi-Timeframe)", "New Pipeline Interface"])

    if mode == "Legacy (Multi-Timeframe)":
        _run_legacy_mode()
    else:
        _run_pipeline_mode()


def _run_legacy_mode():
    """Original multi-timeframe training controller."""
    if not _LEGACY_IMPORTS_OK:
        st.error(f"Legacy imports failed: {_LEGACY_IMPORT_ERROR}")
        return

    data_dir = Path("data")
    if not data_dir.exists():
        data_dir = Path("Data")
    csv_files = sorted([f for f in data_dir.glob("gold_*.csv")]) if data_dir.exists() else []

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
        try:
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
            st.subheader("Strategy Evaluation")
            st.pyplot(plot_equity_curve(backtest_df, plot_dir=plot_dir, model_name="dashboard"))
            st.pyplot(plot_predictions(df, plot_dir=plot_dir, model_name="dashboard"))
            st.pyplot(plot_confusion(df, plot_dir=plot_dir, model_name="dashboard"))
            st.pyplot(plot_rolling_sharpe(backtest_df, plot_dir=plot_dir, model_name="dashboard"))

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.code(traceback.format_exc())


def _run_pipeline_mode():
    """New pipeline-based training controller."""
    if not _PIPELINE_IMPORTS_OK:
        st.error(f"Pipeline imports failed: {_PIPELINE_IMPORT_ERROR}")
        return

    registry = _get_pipeline_registry()
    if not registry:
        st.warning("No pipeline classes available. Check that the pipelines/ package is installed.")
        return

    selected = st.selectbox("Select pipeline:", sorted(registry.keys()))

    if st.button("Train & Evaluate"):
        try:
            with st.spinner("Loading data..."):
                data_splits = load_gold_data()

            pipeline = registry[selected]()

            with st.spinner(f"Training {pipeline.name}..."):
                train_metrics = pipeline.train(data_splits['train'], val_data=data_splits['val'])
                st.success(f"Training complete: {train_metrics}")

            with st.spinner("Evaluating..."):
                eval_metrics = pipeline.evaluate(data_splits['test'])
                st.subheader("Test Evaluation")
                st.json(eval_metrics)

            with st.spinner("Backtesting..."):
                output = pipeline.predict(data_splits['test'])
                test_df = data_splits['test']
                price_col = 'Close' if 'Close' in test_df.columns else test_df.columns[0]
                engine = BacktestEngine()
                result = engine.run(test_df[price_col], output.signals, output.confidence)
                st.subheader("Backtest Metrics")
                st.json(result.metrics)

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.code(traceback.format_exc())


if __name__ == '__main__':
    main()
