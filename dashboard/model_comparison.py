# dashboard/model_comparison.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix


# Load backtest results
def load_backtest(file_path):
    df = pd.read_csv(file_path)
    sharpe = df['strategy_return'].mean() / df['strategy_return'].std() * (252 * 6) ** 0.5
    total_return = df['equity_curve'].iloc[-1] - 1
    accuracy = (df['prediction'] == df[
        'target']).mean() if 'prediction' in df.columns and 'target' in df.columns else None
    return df, accuracy, sharpe, total_return


# Display equity curve
def plot_equity(df, model_name):
    fig, ax = plt.subplots()
    ax.plot(df['equity_curve'], label='Equity Curve')
    ax.set_title(f"{model_name} Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True)
    ax.legend()
    return fig


# Main dashboard app
def main():
    st.set_page_config(page_title="Model Comparison Dashboard", layout="wide")
    st.title("📊 Single vs Multi-Timeframe Model Comparison")

    data_dir = Path("data")

    model_names = ["XGBoost", "LSTM", "CNN-LSTM", "TCN", "Transformer"]
    results = []

    for model in model_names:
        single_path = data_dir / f"{model.lower().replace('-', '_')}_backtest_results.csv"
        multi_path = data_dir / f"{model.lower().replace('-', '_')}_mtf_backtest_results.csv"

        acc_s, sharpe_s, ret_s = None, None, None
        acc_m, sharpe_m, ret_m = None, None, None

        if single_path.exists():
            _, acc_s, sharpe_s, ret_s = load_backtest(single_path)

        if multi_path.exists():
            _, acc_m, sharpe_m, ret_m = load_backtest(multi_path)

        results.append({
            "Model": model,
            "Accuracy (S)": f"{acc_s:.2%}" if acc_s is not None else "-",
            "Accuracy (MTF)": f"{acc_m:.2%}" if acc_m is not None else "-",
            "Sharpe (S)": f"{sharpe_s:.2f}" if sharpe_s is not None else "-",
            "Sharpe (MTF)": f"{sharpe_m:.2f}" if sharpe_m is not None else "-",
            "Return (S)": f"{ret_s:.2%}" if ret_s is not None else "-",
            "Return (MTF)": f"{ret_m:.2%}" if ret_m is not None else "-",
        })

    st.dataframe(pd.DataFrame(results))


if __name__ == "__main__":
    main()
