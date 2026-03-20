# main.py

import streamlit as st
from pathlib import Path
import sys

# Ensure project root is on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Page navigation setup
st.set_page_config(page_title="Gold Trading AI Dashboard", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:", [
    "Train & Evaluate (Legacy)",
    "Pipeline Training",
    "Market Regime Detection",
    "Signal Combiner",
    "Strategy Statistics",
    "Model Comparison",
])

# Route to selected page
if page == "Train & Evaluate (Legacy)":
    from pages.train import run as train_page
    train_page()

elif page == "Pipeline Training":
    from pages.pipeline_train import run as pipeline_train_page
    pipeline_train_page()

elif page == "Market Regime Detection":
    from pages.regime import run as regime_page
    regime_page()

elif page == "Signal Combiner":
    from pages.combiner import run as combiner_page
    combiner_page()

elif page == "Strategy Statistics":
    from pages.statistics import run as statistics_page
    statistics_page()

elif page == "Model Comparison":
    from pages.comparison import run as comparison_page
    comparison_page()
