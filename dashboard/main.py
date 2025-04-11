# main.py

import streamlit as st
from pathlib import Path

# Page navigation setup
st.set_page_config(page_title="📊 Gold Trading AI", layout="wide")

st.sidebar.title("🔍 Navigation")
page = st.sidebar.selectbox("Choose a section:", [
    "📈 Train & Evaluate",
    "📊 Strategy Statistics",
    "🤖 Model Comparison",
])

# Route to selected page
if page == "📈 Train & Evaluate":
    from pages.train import run as train_page

    train_page()

elif page == "📊 Strategy Statistics":
    from pages.statistics import run as statistics_page

    statistics_page()

elif page == "🤖 Model Comparison":
    from pages.comparison import run as comparison_page

    comparison_page()
