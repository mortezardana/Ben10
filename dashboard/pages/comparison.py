# pages/comparison.py
"""Model comparison page — supports both legacy CSV predictions and
new TradingPipeline outputs stored in session state.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))


def _compare_csv_predictions(data_dir):
    """Original CSV-based comparison."""
    prediction_files = list(data_dir.glob("*predictions*.csv"))

    if not prediction_files:
        st.warning("No prediction files found in the data directory.")
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
        st.subheader("Accuracy Comparison")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        colors = plt.cm.Set2(np.linspace(0, 1, len(comparison_df)))
        ax.barh(comparison_df['Model'], comparison_df['Accuracy'], color=colors)
        ax.set_xlabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        ax.set_xlim(0, 1)
        for i, (acc, model) in enumerate(zip(comparison_df['Accuracy'], comparison_df['Model'])):
            ax.text(acc + 0.01, i, f"{acc:.4f}", va='center')
        ax.grid(True, alpha=0.2, axis='x')
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No compatible files with 'target' and 'prediction' columns found.")


def _compare_pipeline_outputs(pipeline_outputs):
    """Compare pipeline outputs stored in session state."""
    if not pipeline_outputs:
        st.info("No pipeline outputs available in session state.")
        return

    st.subheader("Pipeline Signal Comparison")

    # Build comparison table
    rows = []
    for name, output in pipeline_outputs.items():
        signals = output.signals
        confidence = output.confidence

        n_long = int((signals == 1).sum())
        n_flat = int((signals == 0).sum())
        n_short = int((signals == -1).sum())
        total = len(signals)
        mean_conf = float(confidence.mean())

        rows.append({
            "Pipeline": name,
            "Long %": f"{n_long / total:.1%}" if total > 0 else "N/A",
            "Flat %": f"{n_flat / total:.1%}" if total > 0 else "N/A",
            "Short %": f"{n_short / total:.1%}" if total > 0 else "N/A",
            "Mean Confidence": f"{mean_conf:.4f}",
            "Total Bars": total,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Signal distribution chart
    fig, ax = plt.subplots(figsize=(10, 4))
    names = [r["Pipeline"] for r in rows]
    long_pcts = [int((pipeline_outputs[n].signals == 1).sum()) / len(pipeline_outputs[n].signals)
                 for n in names]
    flat_pcts = [int((pipeline_outputs[n].signals == 0).sum()) / len(pipeline_outputs[n].signals)
                 for n in names]
    short_pcts = [int((pipeline_outputs[n].signals == -1).sum()) / len(pipeline_outputs[n].signals)
                  for n in names]

    x = np.arange(len(names))
    width = 0.25
    ax.bar(x - width, long_pcts, width, label='Long', color='#2ecc71')
    ax.bar(x, flat_pcts, width, label='Flat', color='#95a5a6')
    ax.bar(x + width, short_pcts, width, label='Short', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("Fraction")
    ax.set_title("Signal Distribution by Pipeline")
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    st.pyplot(fig)

    # Compare accuracy if actuals available
    if 'last_test_df' in st.session_state and 'target' in st.session_state['last_test_df'].columns:
        test_df = st.session_state['last_test_df']
        actuals = test_df['target']
        # Convert 0/1 target to -1/+1 direction
        actual_direction = actuals.map({0: -1, 1: 1})

        st.subheader("Directional Accuracy (vs Actuals)")
        acc_rows = []
        for name, output in pipeline_outputs.items():
            common = output.signals.index.intersection(actual_direction.index)
            if len(common) > 0:
                # For accuracy, map signals: long(1)->1, short(-1)->-1, flat(0)->0
                # Compare only non-flat signals
                sig = output.signals.reindex(common)
                act = actual_direction.reindex(common)
                non_flat = sig != 0
                if non_flat.any():
                    acc = float((sig[non_flat] == act[non_flat]).mean())
                else:
                    acc = 0.0
                total_acc = float((sig == act).mean())
                acc_rows.append({
                    "Pipeline": name,
                    "Directional Accuracy (excl flat)": f"{acc:.4f}",
                    "Overall Match Rate": f"{total_acc:.4f}",
                    "Non-flat Signals": int(non_flat.sum()),
                })

        if acc_rows:
            st.dataframe(pd.DataFrame(acc_rows), use_container_width=True, hide_index=True)

    # Confidence comparison
    st.subheader("Confidence Distribution Comparison")
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, output in pipeline_outputs.items():
        ax.hist(output.confidence.values, bins=50, alpha=0.4, label=name)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Score Distributions")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)


def run():
    st.title("Model Comparison Dashboard")
    st.markdown("Compare model predictions and metrics across different approaches.")

    # Check for pipeline outputs in session state
    has_pipeline_outputs = 'combiner_pipeline_outputs' in st.session_state
    has_single_output = 'last_pipeline_output' in st.session_state

    sources = ["CSV Prediction Files"]
    if has_pipeline_outputs:
        sources.append("Pipeline Outputs (from Combiner)")
    if has_single_output:
        sources.append("Last Pipeline Run")

    source = st.radio("Comparison source:", sources, horizontal=True)

    if source == "CSV Prediction Files":
        data_dir = Path("data")
        if not data_dir.exists():
            data_dir = Path("Data")
        if data_dir.exists():
            _compare_csv_predictions(data_dir)
        else:
            st.warning("No data directory found.")

    elif source == "Pipeline Outputs (from Combiner)":
        _compare_pipeline_outputs(st.session_state['combiner_pipeline_outputs'])

    elif source == "Last Pipeline Run":
        output = st.session_state['last_pipeline_output']
        name = st.session_state.get('last_pipeline_name', 'unknown')
        _compare_pipeline_outputs({name: output})
