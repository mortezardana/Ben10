# pages/pipeline_train.py
"""Train & evaluate models using the new TradingPipeline interface."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import traceback

sys.path.append(str(Path(__file__).resolve().parents[2]))

from shared.interfaces import TradingPipeline, PipelineOutput
from shared.data_loader import load_gold_data
from shared.config import load_config, AppConfig
from pipeline.backtest import BacktestEngine


# Registry of available pipeline classes
PIPELINE_REGISTRY = {}


def _register_pipelines():
    """Lazily import and register available pipeline classes."""
    global PIPELINE_REGISTRY
    if PIPELINE_REGISTRY:
        return

    try:
        from pipelines.gradient_boosting.xgboost_pipeline import XGBoostPipeline
        PIPELINE_REGISTRY["XGBoost"] = XGBoostPipeline
    except ImportError:
        pass

    try:
        from pipelines.gradient_boosting.lightgbm_pipeline import LightGBMPipeline
        PIPELINE_REGISTRY["LightGBM"] = LightGBMPipeline
    except ImportError:
        pass

    try:
        from pipelines.gradient_boosting.catboost_pipeline import CatBoostPipeline
        PIPELINE_REGISTRY["CatBoost"] = CatBoostPipeline
    except ImportError:
        pass

    try:
        from pipelines.deep_learning.lstm_pipeline import LSTMPipeline
        PIPELINE_REGISTRY["LSTM"] = LSTMPipeline
    except ImportError:
        pass

    try:
        from pipelines.deep_learning.gru_pipeline import GRUPipeline
        PIPELINE_REGISTRY["GRU"] = GRUPipeline
    except ImportError:
        pass

    try:
        from pipelines.deep_learning.cnn_lstm_pipeline import CNNLSTMPipeline
        PIPELINE_REGISTRY["CNN-LSTM"] = CNNLSTMPipeline
    except ImportError:
        pass

    try:
        from pipelines.deep_learning.tcn_pipeline import TCNPipeline
        PIPELINE_REGISTRY["TCN"] = TCNPipeline
    except ImportError:
        pass

    try:
        from pipelines.deep_learning.transformer_pipeline import TransformerPipeline
        PIPELINE_REGISTRY["Transformer"] = TransformerPipeline
    except ImportError:
        pass

    try:
        from pipelines.deep_learning.tabnet_pipeline import TabNetPipeline
        PIPELINE_REGISTRY["TabNet"] = TabNetPipeline
    except ImportError:
        pass

    try:
        from pipelines.deep_learning.crnn_pipeline import CRNNPipeline
        PIPELINE_REGISTRY["CRNN"] = CRNNPipeline
    except ImportError:
        pass

    try:
        from pipelines.deep_learning.tft_pipeline import TFTPipeline
        PIPELINE_REGISTRY["TFT"] = TFTPipeline
    except ImportError:
        pass


def _plot_equity_curve(backtest_result):
    """Plot equity curve from BacktestResult."""
    fig, ax = plt.subplots(figsize=(12, 5))
    equity = backtest_result.equity_curve
    ax.plot(equity.values, label="Strategy Equity", linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_signals(output, prices):
    """Plot signals overlaid on price."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    common_idx = prices.index.intersection(output.signals.index)
    p = prices.reindex(common_idx)
    s = output.signals.reindex(common_idx)
    c = output.confidence.reindex(common_idx)

    ax1.plot(range(len(p)), p.values, color='gray', linewidth=0.8, label='Price')

    long_mask = s == 1
    short_mask = s == -1
    if long_mask.any():
        idx_long = np.where(long_mask.values)[0]
        ax1.scatter(idx_long, p.values[idx_long], color='green', marker='^',
                    s=10, alpha=0.6, label='Long')
    if short_mask.any():
        idx_short = np.where(short_mask.values)[0]
        ax1.scatter(idx_short, p.values[idx_short], color='red', marker='v',
                    s=10, alpha=0.6, label='Short')

    ax1.set_title("Signals on Price")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(c)), c.values * s.values, color=['green' if v > 0 else 'red' if v < 0 else 'gray'
                                                         for v in (c.values * s.values)], alpha=0.6)
    ax2.set_title("Confidence (directional)")
    ax2.set_ylabel("Confidence")
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _plot_rolling_metrics(backtest_result, window=100):
    """Plot rolling Sharpe and drawdown."""
    returns = backtest_result.returns
    equity = backtest_result.equity_curve

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Rolling Sharpe
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(1512)).fillna(0)
    ax1.plot(range(len(rolling_sharpe)), rolling_sharpe.values, linewidth=0.8)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_title(f"Rolling Sharpe Ratio (window={window})")
    ax1.set_ylabel("Sharpe")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    ax2.fill_between(range(len(drawdown)), drawdown.values, 0, color='red', alpha=0.3)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Bar")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _display_metrics(metrics):
    """Display metrics dict in a nice table."""
    if not metrics:
        st.info("No metrics available.")
        return

    rows = []
    for k, v in metrics.items():
        if isinstance(v, float):
            if 'rate' in k or 'return' in k or 'drawdown' in k:
                rows.append({"Metric": k, "Value": f"{v:.4f} ({v:.2%})"})
            else:
                rows.append({"Metric": k, "Value": f"{v:.4f}"})
        elif isinstance(v, tuple):
            rows.append({"Metric": k, "Value": f"({v[0]:.6f}, {v[1]:.6f})"})
        else:
            rows.append({"Metric": k, "Value": str(v)})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def run():
    st.title("Pipeline Training & Evaluation")
    st.markdown(
        "Train models using the standardized `TradingPipeline` interface. "
        "Each pipeline produces calibrated signals and confidence scores."
    )

    _register_pipelines()

    if not PIPELINE_REGISTRY:
        st.error(
            "No pipeline classes could be imported. "
            "Check that the `pipelines/` package is on PYTHONPATH."
        )
        return

    # --- Sidebar controls ---
    st.sidebar.subheader("Pipeline Settings")
    available_names = sorted(PIPELINE_REGISTRY.keys())
    selected_pipeline = st.sidebar.selectbox("Select pipeline:", available_names)

    # Data loading options
    st.sidebar.subheader("Data Settings")
    data_dir = Path("Data")
    if not data_dir.exists():
        data_dir = Path("data")
    csv_files = sorted(data_dir.glob("gold_*.csv")) if data_dir.exists() else []

    use_shared_loader = st.sidebar.checkbox("Use shared data loader (recommended)", value=True)

    if not use_shared_loader:
        if csv_files:
            selected_data = st.sidebar.selectbox("Data file:", csv_files)
        else:
            st.sidebar.warning("No gold_*.csv files found.")
            return

    # Backtest settings
    st.sidebar.subheader("Backtest Settings")
    initial_capital = st.sidebar.number_input("Initial capital ($)", value=100000, step=10000)
    commission = st.sidebar.number_input("Commission (bps)", value=1.0, step=0.5) / 10000
    slippage = st.sidebar.number_input("Slippage (bps)", value=1.0, step=0.5) / 10000

    # --- Main area ---
    run_btn = st.button("Run Pipeline")

    if run_btn:
        with st.spinner(f"Loading data and training {selected_pipeline} pipeline..."):
            try:
                # Load data
                if use_shared_loader:
                    data_splits = load_gold_data()
                    train_df = data_splits['train']
                    val_df = data_splits['val']
                    test_df = data_splits['test']
                    full_df = data_splits['full_df']
                else:
                    from pipeline.data_loader import load_data, normalize_features
                    full_df = load_data(str(selected_data))
                    n = len(full_df)
                    train_n = int(n * 0.7)
                    val_n = int(n * 0.1)
                    full_df, _ = normalize_features(full_df, train_end_idx=train_n)
                    train_df = full_df.iloc[:train_n]
                    val_df = full_df.iloc[train_n:train_n + val_n]
                    test_df = full_df.iloc[train_n + val_n:]

                st.success(f"Data loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

            except Exception as e:
                st.error(f"Data loading failed: {e}")
                st.code(traceback.format_exc())
                return

        # Instantiate pipeline
        pipeline_cls = PIPELINE_REGISTRY[selected_pipeline]
        pipeline = pipeline_cls()

        # Train
        with st.spinner(f"Training {selected_pipeline}..."):
            try:
                train_metrics = pipeline.train(train_df, val_data=val_df)
                st.success(f"Training complete for {pipeline.name}")
                st.subheader("Training Metrics")
                _display_metrics(train_metrics)
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.code(traceback.format_exc())
                return

        # Evaluate
        with st.spinner("Evaluating on test set..."):
            try:
                eval_metrics = pipeline.evaluate(test_df)
                st.subheader("Test Set Evaluation")
                _display_metrics(eval_metrics)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.code(traceback.format_exc())
                return

        # Generate predictions
        with st.spinner("Generating signals..."):
            try:
                output = pipeline.predict(test_df)
                st.subheader("Signal Distribution")
                signal_counts = output.signals.value_counts().sort_index()
                col1, col2, col3 = st.columns(3)
                col1.metric("Long signals", int(signal_counts.get(1, 0)))
                col2.metric("Flat signals", int(signal_counts.get(0, 0)))
                col3.metric("Short signals", int(signal_counts.get(-1, 0)))

                st.write(f"Mean confidence: {output.confidence.mean():.4f}")

                # Store output in session state for other pages
                st.session_state['last_pipeline_output'] = output
                st.session_state['last_pipeline_name'] = pipeline.name
                st.session_state['last_test_df'] = test_df

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.code(traceback.format_exc())
                return

        # Backtest
        with st.spinner("Running backtest..."):
            try:
                # Get price column
                price_col = 'Close'
                if price_col not in test_df.columns:
                    price_candidates = [c for c in test_df.columns if 'close' in c.lower()]
                    price_col = price_candidates[0] if price_candidates else test_df.columns[0]

                engine = BacktestEngine(
                    initial_capital=initial_capital,
                    commission=commission,
                    slippage=slippage,
                )
                backtest_result = engine.run(
                    prices=test_df[price_col],
                    signals=output.signals,
                    confidence=output.confidence,
                )

                st.subheader("Backtest Results")
                _display_metrics(backtest_result.metrics)

                # Plots
                st.subheader("Equity Curve")
                st.pyplot(_plot_equity_curve(backtest_result))

                st.subheader("Signals on Price")
                st.pyplot(_plot_signals(output, test_df[price_col]))

                st.subheader("Rolling Metrics & Drawdown")
                st.pyplot(_plot_rolling_metrics(backtest_result))

                # Trade log
                if not backtest_result.trades.empty:
                    st.subheader("Recent Trades")
                    st.dataframe(backtest_result.trades.tail(20), use_container_width=True)

                # Store for other pages
                st.session_state['last_backtest_result'] = backtest_result

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.code(traceback.format_exc())
