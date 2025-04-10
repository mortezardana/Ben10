# dashboard/model_comparison.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Load backtest results
def load_backtest(file_path):
    df = pd.read_csv(file_path)
    sharpe = df['strategy_return'].mean() / df['strategy_return'].std() * (252 * 6) ** 0.5
    total_return = df['equity_curve'].iloc[-1] - 1
    return df, sharpe, total_return


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


# Display confusion matrix
def plot_confusion(df, model_name):
    cm = confusion_matrix(df['target'], df['prediction'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title(f"{model_name} Confusion Matrix")
    return fig


# Main dashboard app
def main():
    st.set_page_config(page_title="Model Comparison Dashboard", layout="wide")
    st.title("📊 Model Comparison Dashboard")

    data_dir = Path("data")

    models = [
        ("XGBoost", data_dir / "backtest_results.csv"),
        ("LSTM", data_dir / "lstm_backtest_results.csv"),
        ("CNN-LSTM", data_dir / "cnn_lstm_backtest_results.csv"),
        ("TCN", data_dir / "tcn_backtest_results.csv")
    ]

    cols = st.columns(len(models))

    for (model_name, path), col in zip(models, cols):
        if path.exists():
            df, sharpe, total_return = load_backtest(path)

            with col:
                st.subheader(model_name)
                if 'prediction' in df.columns and 'target' in df.columns:
                    accuracy = (df['prediction'] == df['target']).mean()
                    st.metric(label="Accuracy", value=f"{accuracy:.2%}")
                else:
                    st.warning("Missing 'prediction' or 'target'")

                st.metric(label="Sharpe Ratio", value=f"{sharpe:.2f}")
                st.metric(label="Total Return", value=f"{total_return:.2%}")

                with st.expander("📈 Equity Curve"):
                    st.pyplot(plot_equity(df, model_name))

                if 'prediction' in df.columns and 'target' in df.columns:
                    with st.expander("🧮 Confusion Matrix"):
                        st.pyplot(plot_confusion(df, model_name))
        else:
            col.warning(f"No data found for {model_name}")


if __name__ == "__main__":
    main()
