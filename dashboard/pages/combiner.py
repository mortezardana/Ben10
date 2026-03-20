# pages/combiner.py
"""Signal combiner visualization: weights, agreement, and combined output."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import traceback

sys.path.append(str(Path(__file__).resolve().parents[2]))

from shared.interfaces import PipelineOutput


def _plot_combiner_weights(weights):
    """Bar chart of combiner weights per pipeline."""
    fig, ax = plt.subplots(figsize=(10, 4))
    names = list(weights.keys())
    vals = list(weights.values())

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.barh(names, vals, color=colors, edgecolor='white')
    ax.set_xlabel("Weight")
    ax.set_title("Combiner Weights by Pipeline")
    ax.set_xlim(0, max(vals) * 1.2 if vals else 1)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va='center', fontsize=9)

    ax.grid(True, alpha=0.2, axis='x')
    fig.tight_layout()
    return fig


def _plot_signal_agreement(pipeline_outputs):
    """Heatmap showing pairwise signal agreement between pipelines."""
    names = list(pipeline_outputs.keys())
    if len(names) < 2:
        return None

    # Build agreement matrix
    n = len(names)
    agreement = np.zeros((n, n))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            si = pipeline_outputs[name_i].signals
            sj = pipeline_outputs[name_j].signals
            common = si.index.intersection(sj.index)
            if len(common) > 0:
                agreement[i, j] = (si.reindex(common) == sj.reindex(common)).mean()
            else:
                agreement[i, j] = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(agreement, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{agreement[i, j]:.2f}", ha='center', va='center',
                    fontsize=10, color='black')

    plt.colorbar(im, ax=ax, label="Agreement Rate")
    ax.set_title("Pairwise Signal Agreement")
    fig.tight_layout()
    return fig


def _plot_signal_correlation(pipeline_outputs):
    """Plot signal correlation matrix."""
    names = list(pipeline_outputs.keys())
    if len(names) < 2:
        return None

    signals_df = pd.DataFrame()
    for name, output in pipeline_outputs.items():
        signals_df[name] = output.signals

    corr = signals_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha='center', va='center',
                    fontsize=10, color='black')

    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Signal Correlation Matrix")
    fig.tight_layout()
    return fig


def _plot_combined_vs_individual(combined_output, pipeline_outputs, window=50):
    """Plot rolling accuracy of combined vs individual signals."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # If we have actuals in session state, compute rolling accuracy
    if 'last_test_df' in st.session_state and 'target' in st.session_state['last_test_df'].columns:
        actuals = st.session_state['last_test_df']['target']
        # Convert target 0/1 to -1/+1 direction for comparison
        actual_direction = actuals.map({0: -1, 1: 1})

        for name, output in pipeline_outputs.items():
            common = output.signals.index.intersection(actual_direction.index)
            if len(common) < window:
                continue
            correct = (output.signals.reindex(common) == actual_direction.reindex(common)).astype(float)
            rolling_acc = correct.rolling(window).mean()
            ax.plot(range(len(rolling_acc)), rolling_acc.values, label=name, alpha=0.6, linewidth=1)

        if combined_output is not None:
            common = combined_output.signals.index.intersection(actual_direction.index)
            if len(common) >= window:
                correct = (combined_output.signals.reindex(common) == actual_direction.reindex(common)).astype(float)
                rolling_acc = correct.rolling(window).mean()
                ax.plot(range(len(rolling_acc)), rolling_acc.values, label="Combined",
                        linewidth=2.5, color='black', linestyle='--')

        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f"Rolling Accuracy (window={window})")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Accuracy")
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No actuals available for accuracy comparison.\n"
                "Run a pipeline from the Pipeline Training page first.",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Rolling Accuracy (no actuals available)")

    fig.tight_layout()
    return fig


def _plot_confidence_distribution(pipeline_outputs, combined_output=None):
    """Plot confidence distributions for each pipeline."""
    n = len(pipeline_outputs) + (1 if combined_output else 0)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]

    colors = plt.cm.Set2(np.linspace(0, 1, n))

    for i, (name, output) in enumerate(pipeline_outputs.items()):
        axes[i].hist(output.confidence.values, bins=50, color=colors[i],
                     alpha=0.7, edgecolor='white')
        axes[i].set_title(f"{name}\nmean={output.confidence.mean():.3f}")
        axes[i].set_xlabel("Confidence")
        axes[i].grid(True, alpha=0.2)

    if combined_output:
        axes[-1].hist(combined_output.confidence.values, bins=50, color='black',
                      alpha=0.6, edgecolor='white')
        axes[-1].set_title(f"Combined\nmean={combined_output.confidence.mean():.3f}")
        axes[-1].set_xlabel("Confidence")
        axes[-1].grid(True, alpha=0.2)

    fig.suptitle("Confidence Score Distributions", y=1.02)
    fig.tight_layout()
    return fig


def run():
    st.title("Signal Combiner Analysis")
    st.markdown(
        "Combine signals from multiple pipelines using weighted voting or stacking. "
        "Visualize weights, signal agreement, and the combined output."
    )

    # --- Mode selection ---
    mode = st.radio("Mode:", ["Interactive (train pipelines now)", "From Session State"], horizontal=True)

    pipeline_outputs = {}
    combined_output = None
    weights = {}

    if mode == "From Session State":
        # Check if we have pipeline outputs stored
        if 'combiner_pipeline_outputs' not in st.session_state:
            st.info(
                "No pipeline outputs found in session state. "
                "Run multiple pipelines from the Pipeline Training page first, "
                "or switch to Interactive mode."
            )
            return
        pipeline_outputs = st.session_state.get('combiner_pipeline_outputs', {})
        combined_output = st.session_state.get('combiner_combined_output', None)
        weights = st.session_state.get('combiner_weights', {})

    else:  # Interactive mode
        st.subheader("Setup Pipelines")

        # Import pipeline registry
        try:
            from pages.pipeline_train import _register_pipelines, PIPELINE_REGISTRY
            _register_pipelines()
        except ImportError:
            st.error("Cannot import pipeline registry.")
            return

        if not PIPELINE_REGISTRY:
            st.warning("No pipeline classes available.")
            return

        available_names = sorted(PIPELINE_REGISTRY.keys())
        selected_pipelines = st.multiselect(
            "Select pipelines to combine (at least 2):", available_names,
            default=available_names[:min(2, len(available_names))]
        )

        if len(selected_pipelines) < 2:
            st.warning("Select at least 2 pipelines to combine.")
            return

        # Combiner settings
        st.sidebar.subheader("Combiner Settings")
        combiner_method = st.sidebar.selectbox("Combiner method:", ["Weighted Voting", "Stacking"])
        lookback = st.sidebar.slider("Lookback window (for weights):", 50, 500, 100, step=10)

        run_btn = st.button("Train & Combine")

        if run_btn:
            # Load data
            with st.spinner("Loading data..."):
                try:
                    from shared.data_loader import load_gold_data
                    data_splits = load_gold_data()
                    train_df = data_splits['train']
                    val_df = data_splits['val']
                    test_df = data_splits['test']
                except Exception as e:
                    st.error(f"Data loading failed: {e}")
                    return

            # Train each pipeline
            trained_pipelines = {}
            progress = st.progress(0)
            for i, name in enumerate(selected_pipelines):
                with st.spinner(f"Training {name}..."):
                    try:
                        pipeline_cls = PIPELINE_REGISTRY[name]
                        pipeline = pipeline_cls()
                        pipeline.train(train_df, val_data=val_df)
                        output = pipeline.predict(test_df)
                        pipeline_outputs[name] = output
                        trained_pipelines[name] = pipeline
                        st.success(f"{name} trained and predicted.")
                    except Exception as e:
                        st.warning(f"Failed to train {name}: {e}")
                progress.progress((i + 1) / len(selected_pipelines))

            if len(pipeline_outputs) < 2:
                st.error("Need at least 2 successful pipelines to combine.")
                return

            # Combine
            with st.spinner("Combining signals..."):
                try:
                    if combiner_method == "Weighted Voting":
                        from combiner.weighted_voting import WeightedVotingCombiner
                        combiner = WeightedVotingCombiner(lookback=lookback)

                        # Use actuals for weight calculation if available
                        actuals = test_df['target'] if 'target' in test_df.columns else None
                        combined_output = combiner.combine(pipeline_outputs, recent_actuals=actuals)
                        weights = combiner.get_weights()

                    else:  # Stacking
                        from combiner.stacking import StackingCombiner
                        combiner = StackingCombiner()

                        # Need OOS predictions on validation set for training
                        val_outputs = {}
                        for name, pipeline in trained_pipelines.items():
                            try:
                                val_outputs[name] = pipeline.predict(val_df)
                            except Exception:
                                pass

                        if len(val_outputs) >= 2:
                            actuals = val_df['target'] if 'target' in val_df.columns else None
                            if actuals is not None:
                                combiner.train(val_outputs, market_features=None, actuals=actuals)
                                combined_output = combiner.combine(pipeline_outputs)
                                weights = {name: 1.0 / len(pipeline_outputs) for name in pipeline_outputs}
                            else:
                                st.error("No target column for stacking training.")
                                return
                        else:
                            st.error("Need at least 2 valid validation outputs for stacking.")
                            return

                    st.success("Signals combined!")

                except Exception as e:
                    st.error(f"Combining failed: {e}")
                    st.code(traceback.format_exc())
                    return

            # Store in session state
            st.session_state['combiner_pipeline_outputs'] = pipeline_outputs
            st.session_state['combiner_combined_output'] = combined_output
            st.session_state['combiner_weights'] = weights
            st.session_state['last_test_df'] = test_df

    # --- Visualizations (shared for both modes) ---
    if not pipeline_outputs:
        return

    st.markdown("---")
    st.header("Combiner Analysis")

    # Weights
    if weights:
        st.subheader("Pipeline Weights")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(_plot_combiner_weights(weights))
        with col2:
            weights_df = pd.DataFrame([
                {"Pipeline": k, "Weight": f"{v:.4f}"}
                for k, v in sorted(weights.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(weights_df, use_container_width=True, hide_index=True)

    # Signal agreement
    if len(pipeline_outputs) >= 2:
        st.subheader("Pairwise Signal Agreement")
        fig = _plot_signal_agreement(pipeline_outputs)
        if fig is not None:
            st.pyplot(fig)

        st.subheader("Signal Correlation")
        fig = _plot_signal_correlation(pipeline_outputs)
        if fig is not None:
            st.pyplot(fig)

    # Confidence distributions
    st.subheader("Confidence Distributions")
    st.pyplot(_plot_confidence_distribution(pipeline_outputs, combined_output))

    # Combined vs individual
    st.subheader("Rolling Accuracy Comparison")
    st.pyplot(_plot_combined_vs_individual(combined_output, pipeline_outputs))

    # Combined signal summary
    if combined_output is not None:
        st.subheader("Combined Signal Summary")
        signal_counts = combined_output.signals.value_counts().sort_index()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Long", int(signal_counts.get(1, 0)))
        col2.metric("Flat", int(signal_counts.get(0, 0)))
        col3.metric("Short", int(signal_counts.get(-1, 0)))
        col4.metric("Mean Confidence", f"{combined_output.confidence.mean():.4f}")

        # Show where pipelines agree/disagree
        st.subheader("Signal Agreement Timeline")
        names = list(pipeline_outputs.keys())
        signals_df = pd.DataFrame({name: pipeline_outputs[name].signals for name in names})
        signals_df['combined'] = combined_output.signals

        # Agreement = fraction of pipelines that agree with the combined signal
        def agreement_score(row):
            combined = row['combined']
            individual = row.drop('combined')
            if combined == 0:
                return (individual == 0).mean()
            return (individual == combined).mean()

        signals_df['agreement'] = signals_df.apply(agreement_score, axis=1)

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.fill_between(range(len(signals_df)), signals_df['agreement'].values,
                        alpha=0.6, color='steelblue')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% agreement')
        ax.set_title("Signal Agreement Over Time (fraction of pipelines agreeing with combined)")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Agreement")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
