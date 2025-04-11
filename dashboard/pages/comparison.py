# pages/comparison.py

import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def run():
    st.title("🤖 Model Comparison Dashboard")
    st.markdown("Compare model predictions and metrics across timeframes and techniques.")

    data_dir = Path("data")
    prediction_files = list(data_dir.glob("*predictions*.csv"))

    if not prediction_files:
        st.warning("No prediction files found in the /data directory.")
        return

    selected_files = st.multiselect("Select prediction files to compare:", prediction_files)

    comparison_df = pd.DataFrame()
    for file in selected_files:
        df = pd.read_csv(file)
        if 'target' not in df.columns or 'prediction' not in df.columns:
            continue
        accuracy = (df['target'] == df['prediction']).mean()
        temp = pd.DataFrame({
            'Model': [file.stem],
            'Accuracy': [accuracy],
            'Total Samples': [len(df)]
        })
        comparison_df = pd.concat([comparison_df, temp], ignore_index=True)

    if not comparison_df.empty:
        st.subheader("📊 Accuracy Comparison")
        st.dataframe(comparison_df)

        fig, ax = plt.subplots()
        ax.barh(comparison_df['Model'], comparison_df['Accuracy'])
        ax.set_xlabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        st.pyplot(fig)
    else:
        st.info("No compatible files with 'target' and 'prediction' columns found.")
