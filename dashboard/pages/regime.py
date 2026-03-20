# pages/regime.py
"""Regime detection visualization using the HMM pipeline."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import sys
import traceback

sys.path.append(str(Path(__file__).resolve().parents[2]))


REGIME_COLORS = {
    'bull': '#2ecc71',
    'choppy': '#f39c12',
    'bear': '#e74c3c',
}

REGIME_LABELS = {
    1: 'bull',
    0: 'choppy',
    -1: 'bear',
}


def _plot_regime_overlay(prices, regimes, title="Price with Regime Overlay"):
    """Plot price chart with background colored by regime."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(range(len(prices)), prices.values, color='black', linewidth=0.8, label='Price')

    # Color background by regime
    for i in range(len(regimes) - 1):
        signal = regimes.iloc[i]
        label = REGIME_LABELS.get(signal, 'choppy')
        color = REGIME_COLORS.get(label, '#cccccc')
        ax.axvspan(i, i + 1, alpha=0.15, color=color, linewidth=0)

    # Legend
    patches = [mpatches.Patch(color=REGIME_COLORS[k], alpha=0.4, label=k.capitalize())
               for k in REGIME_COLORS]
    ax.legend(handles=patches, loc='upper left')
    ax.set_title(title)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def _plot_regime_posteriors(posteriors_df):
    """Plot posterior probabilities over time."""
    fig, ax = plt.subplots(figsize=(14, 4))

    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    labels = ['Bull prob', 'Choppy prob', 'Bear prob']

    bottom = np.zeros(len(posteriors_df))
    for i, col in enumerate(posteriors_df.columns):
        vals = posteriors_df[col].values
        ax.fill_between(range(len(posteriors_df)), bottom, bottom + vals,
                        color=colors[i % len(colors)], alpha=0.6, label=labels[i % len(labels)])
        bottom += vals

    ax.set_title("Regime Posterior Probabilities (Stacked)")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def _plot_regime_transitions(regimes):
    """Plot regime state transitions over time."""
    fig, ax = plt.subplots(figsize=(14, 3))

    signal_vals = regimes.values
    colors = []
    for v in signal_vals:
        label = REGIME_LABELS.get(v, 'choppy')
        colors.append(REGIME_COLORS.get(label, '#cccccc'))

    ax.bar(range(len(signal_vals)), signal_vals, color=colors, width=1.0, alpha=0.7)
    ax.set_title("Regime State Over Time")
    ax.set_xlabel("Bar")
    ax.set_ylabel("State (-1=Bear, 0=Choppy, 1=Bull)")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Bear", "Choppy", "Bull"])
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    return fig


def _plot_regime_statistics(prices, regimes):
    """Plot per-regime return distributions."""
    returns = prices.pct_change().dropna()
    common_idx = returns.index.intersection(regimes.index)
    returns = returns.reindex(common_idx)
    regimes_aligned = regimes.reindex(common_idx)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (signal_val, regime_name) in zip(axes, REGIME_LABELS.items()):
        mask = regimes_aligned == signal_val
        regime_returns = returns[mask]

        if len(regime_returns) > 0:
            ax.hist(regime_returns.values, bins=50, color=REGIME_COLORS.get(regime_name, '#ccc'),
                    alpha=0.7, edgecolor='white')
            mean_r = regime_returns.mean()
            std_r = regime_returns.std()
            ax.axvline(mean_r, color='black', linestyle='--', linewidth=1.5,
                       label=f"Mean: {mean_r:.5f}")
            ax.set_title(f"{regime_name.capitalize()} (n={len(regime_returns)})")
            ax.set_xlabel("Return")
            ax.legend(fontsize=8)
        else:
            ax.set_title(f"{regime_name.capitalize()} (n=0)")
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)

        ax.grid(True, alpha=0.2)

    fig.suptitle("Return Distribution by Regime", y=1.02)
    fig.tight_layout()
    return fig


def run():
    st.title("Market Regime Detection")
    st.markdown(
        "Visualize market regimes detected by the HMM (Hidden Markov Model) pipeline. "
        "The model classifies each bar into one of three states: **Bull**, **Choppy**, or **Bear**."
    )

    # Try to import the HMM pipeline
    try:
        from pipelines.regime.hmm_pipeline import HMMRegimePipeline
    except ImportError as e:
        st.error(
            f"Cannot import HMMRegimePipeline: {e}. "
            "Make sure `hmmlearn` is installed: `pip install hmmlearn`"
        )
        return

    # Data loading
    st.sidebar.subheader("Data Settings")

    use_shared_loader = st.sidebar.checkbox("Use shared data loader", value=True)

    data = None
    prices = None

    if use_shared_loader:
        try:
            from shared.data_loader import load_gold_data
            data_splits = load_gold_data()
            # Use full dataset for regime detection
            data = data_splits['full_df']
            prices = data['Close'] if 'Close' in data.columns else None
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return
    else:
        data_dir = Path("Data")
        if not data_dir.exists():
            data_dir = Path("data")
        csv_files = sorted(data_dir.glob("gold_*.csv")) if data_dir.exists() else []

        if not csv_files:
            st.warning("No gold_*.csv files found.")
            return

        selected_file = st.sidebar.selectbox("Data file:", csv_files)
        try:
            data = pd.read_csv(selected_file)
            if 'Close' in data.columns:
                prices = data['Close']
            else:
                close_candidates = [c for c in data.columns if 'close' in c.lower()]
                if close_candidates:
                    prices = data[close_candidates[0]]
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

    if data is None or prices is None:
        st.warning("Could not find price data. Ensure a 'Close' column exists.")
        return

    # HMM settings
    st.sidebar.subheader("HMM Settings")
    n_components = st.sidebar.slider("Number of states", min_value=2, max_value=5, value=3)
    n_iter = st.sidebar.slider("EM iterations", min_value=10, max_value=500, value=100, step=10)

    run_btn = st.button("Detect Regimes")

    if run_btn:
        with st.spinner("Training HMM regime model..."):
            try:
                hmm = HMMRegimePipeline(config={
                    'n_components': n_components,
                    'n_iter': n_iter,
                    'random_state': 42,
                })
                train_metrics = hmm.train(data)

                st.success("HMM training complete!")

                # Display state mapping
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("State Mapping")
                    state_map_df = pd.DataFrame([
                        {"HMM State": k, "Regime": v}
                        for k, v in train_metrics.get('state_map', {}).items()
                    ])
                    st.dataframe(state_map_df, use_container_width=True, hide_index=True)

                with col2:
                    st.subheader("State Mean Returns")
                    returns_df = pd.DataFrame([
                        {"HMM State": k, "Mean Return": f"{v:.6f}"}
                        for k, v in train_metrics.get('state_returns', {}).items()
                    ])
                    st.dataframe(returns_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"HMM training failed: {e}")
                st.code(traceback.format_exc())
                return

        # Generate predictions
        with st.spinner("Generating regime predictions..."):
            try:
                output = hmm.predict(data)
                regime_features = hmm.get_regime_features(data)

                # Store in session state
                st.session_state['regime_output'] = output
                st.session_state['regime_features'] = regime_features
                st.session_state['regime_model'] = hmm

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.code(traceback.format_exc())
                return

        # Visualizations
        st.subheader("Regime Overlay on Price")
        st.pyplot(_plot_regime_overlay(prices, output.signals))

        st.subheader("Regime State Transitions")
        st.pyplot(_plot_regime_transitions(output.signals))

        # Posterior probabilities
        posterior_cols = [c for c in regime_features.columns if c.startswith('regime_prob_')]
        if posterior_cols:
            st.subheader("Regime Posterior Probabilities")
            st.pyplot(_plot_regime_posteriors(regime_features[posterior_cols]))

        # Per-regime statistics
        st.subheader("Return Distributions by Regime")
        st.pyplot(_plot_regime_statistics(prices, output.signals))

        # Summary statistics
        st.subheader("Regime Summary")
        summary_rows = []
        returns = prices.pct_change().dropna()
        common_idx = returns.index.intersection(output.signals.index)
        for signal_val, regime_name in REGIME_LABELS.items():
            mask = output.signals.reindex(common_idx) == signal_val
            regime_returns = returns.reindex(common_idx)[mask]
            n_bars = int(mask.sum())
            pct = n_bars / len(common_idx) * 100 if len(common_idx) > 0 else 0
            mean_ret = float(regime_returns.mean()) if len(regime_returns) > 0 else 0
            std_ret = float(regime_returns.std()) if len(regime_returns) > 0 else 0
            mean_conf = float(output.confidence.reindex(common_idx)[mask].mean()) if n_bars > 0 else 0
            summary_rows.append({
                "Regime": regime_name.capitalize(),
                "Bars": n_bars,
                "% of Time": f"{pct:.1f}%",
                "Mean Return": f"{mean_ret:.6f}",
                "Std Return": f"{std_ret:.6f}",
                "Mean Confidence": f"{mean_conf:.4f}",
            })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
