# pages/statistics.py

import streamlit as st
import pandas as pd
from pathlib import Path


def run():
    st.title("📊 Strategy Statistics")

    st.markdown("This page will provide historical backtest performance and performance summaries.")

    results_dir = Path("data")
    backtest_files = list(results_dir.glob("*_backtest_results.csv"))

    if not backtest_files:
        st.warning("No backtest result files found in the /data folder.")
        return

    selected_file = st.selectbox("Select a backtest result file:", backtest_files)

    if selected_file:
        df = pd.read_csv(selected_file)
        st.subheader(f"Preview: {selected_file.name}")
        st.dataframe(df.tail())

        if 'equity_curve' in df.columns:
            st.line_chart(df['equity_curve'])

        st.markdown("### Key Statistics")
        st.write("Total Trades:", df.shape[0])
        st.write("Win Rate:", f"{(df['strategy_return'] > 0).mean():.2%}")
        st.write("Cumulative Return:", f"{(df['strategy_return'] + 1).prod() - 1:.2%}")
        st.write("Sharpe Ratio (Rolling):", df['sharpe_ratio'].mean() if 'sharpe_ratio' in df.columns else "N/A")
