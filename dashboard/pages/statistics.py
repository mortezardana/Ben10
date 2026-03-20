# pages/statistics.py
"""Strategy statistics page — supports both legacy CSV backtests and
the new BacktestResult / BacktestEngine output.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))


def _display_legacy_stats(df, filename):
    """Display stats from a legacy backtest CSV."""
    st.subheader(f"Preview: {filename}")
    st.dataframe(df.tail(10), use_container_width=True)

    if 'equity_curve' in df.columns:
        st.line_chart(df['equity_curve'])

    st.markdown("### Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", df.shape[0])

    if 'strategy_return' in df.columns:
        win_rate = (df['strategy_return'] > 0).mean()
        cum_return = (df['strategy_return'] + 1).prod() - 1
        col2.metric("Win Rate", f"{win_rate:.2%}")
        col3.metric("Cumulative Return", f"{cum_return:.2%}")
    else:
        col2.metric("Win Rate", "N/A")
        col3.metric("Cumulative Return", "N/A")

    if 'sharpe_ratio' in df.columns:
        col4.metric("Mean Sharpe Ratio", f"{df['sharpe_ratio'].mean():.4f}")
    else:
        col4.metric("Mean Sharpe Ratio", "N/A")


def _display_pipeline_backtest_stats(backtest_result):
    """Display stats from a new-style BacktestResult object."""
    st.subheader("Pipeline Backtest Results")

    metrics = backtest_result.metrics
    if metrics:
        # Key metrics at a glance
        cols = st.columns(4)
        cols[0].metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.4f}")
        cols[1].metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
        cols[2].metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
        cols[3].metric("Total Return", f"{metrics.get('total_return', 0):.2%}")

        # Full metrics table
        st.markdown("### Detailed Metrics")
        rows = []
        for k, v in metrics.items():
            if isinstance(v, float):
                rows.append({"Metric": k, "Value": f"{v:.6f}"})
            elif isinstance(v, tuple):
                rows.append({"Metric": k, "Value": f"({v[0]:.6f}, {v[1]:.6f})"})
            else:
                rows.append({"Metric": k, "Value": str(v)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Equity curve
    if backtest_result.equity_curve is not None and len(backtest_result.equity_curve) > 0:
        st.markdown("### Equity Curve")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(backtest_result.equity_curve.values, linewidth=1)
        ax.set_title("Equity Curve")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

    # Drawdown
    if backtest_result.equity_curve is not None and len(backtest_result.equity_curve) > 1:
        st.markdown("### Drawdown")
        equity = backtest_result.equity_curve
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.fill_between(range(len(drawdown)), drawdown.values, 0, color='red', alpha=0.3)
        ax.set_title("Drawdown Over Time")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)

    # Trade summary
    trades = backtest_result.trades
    if trades is not None and not trades.empty:
        st.markdown("### Trade Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trades", len(trades))

        if 'pnl' in trades.columns:
            col2.metric("Avg PnL", f"{trades['pnl'].mean():.6f}")
            col3.metric("Best Trade", f"{trades['pnl'].max():.6f}")

            # PnL distribution
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(trades['pnl'].values, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
            ax.axvline(0, color='red', linestyle='--', linewidth=1)
            ax.set_title("Trade PnL Distribution")
            ax.set_xlabel("PnL")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)

        if 'duration' in trades.columns:
            st.markdown("### Trade Duration Distribution")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(trades['duration'].values, bins=30, color='teal', alpha=0.7, edgecolor='white')
            ax.set_title("Trade Duration (bars)")
            ax.set_xlabel("Duration")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("### Recent Trades")
        st.dataframe(trades.tail(20), use_container_width=True)


def run():
    st.title("Strategy Statistics")

    # Check for pipeline backtest in session state
    has_pipeline_result = 'last_backtest_result' in st.session_state

    source = st.radio(
        "Data source:",
        ["CSV backtest files", "Pipeline backtest (session)"] if has_pipeline_result else ["CSV backtest files"],
        horizontal=True,
    )

    if source == "Pipeline backtest (session)" and has_pipeline_result:
        _display_pipeline_backtest_stats(st.session_state['last_backtest_result'])
        return

    # Legacy CSV-based statistics
    st.markdown("Load and inspect backtest result CSV files from the `data/` directory.")

    results_dir = Path("data")
    if not results_dir.exists():
        results_dir = Path("Data")

    backtest_files = list(results_dir.glob("*_backtest_results.csv")) if results_dir.exists() else []

    if not backtest_files:
        st.warning("No backtest result files found in the data folder.")
        if has_pipeline_result:
            st.info("However, a pipeline backtest result is available in session state. "
                    "Select 'Pipeline backtest (session)' above to view it.")
        return

    selected_file = st.selectbox("Select a backtest result file:", backtest_files)

    if selected_file:
        try:
            df = pd.read_csv(selected_file)
            _display_legacy_stats(df, selected_file.name)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
