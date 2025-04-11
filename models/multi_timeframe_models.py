# models/multi_timeframe_models.py

import pandas as pd
from models.baseline_ml import train_baseline_ml
from models.lstm_model import train_lstm_model, predict_lstm
from models.predictor import predict, load_model
from utils.logger import get_logger
from pathlib import Path
import joblib
import streamlit as st

logger = get_logger("multi_timeframe")


def train_models_per_timeframe(df: pd.DataFrame, timeframes: list, target_col: str = 'target',
                               save_dir: Path = Path("models")) -> dict:
    """
    Train a baseline ML model per timeframe using features with that timeframe's suffix.
    """
    trained_models = {}

    for tf in timeframes:
        tf_cols = [col for col in df.columns if col.endswith(f"_{tf}")]
        tf_df = df[tf_cols].copy()
        tf_df[target_col] = df[target_col].values  # include target

        logger.info(f"Training model for timeframe: {tf}, features: {len(tf_cols)}")
        model, metrics = train_baseline_ml(tf_df, target_col=target_col, save_model=False)

        model_path = save_dir / f"xgb_{tf}.pkl"
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)

        trained_models[tf] = model

    return trained_models


def predict_models_per_timeframe(df: pd.DataFrame, timeframes: list, save_dir: Path = Path("models")) -> pd.DataFrame:
    """
    Generate predictions using pre-trained models per timeframe. Retrain if missing.
    """
    df_preds = df.copy()

    for tf in timeframes:
        tf_cols = [col for col in df.columns if col.endswith(f"_{tf}")]
        tf_df = df_preds[tf_cols].copy()
        tf_df['target'] = df['target'].values

        model_path = save_dir / f"xgb_{tf}.pkl"

        if not model_path.exists():
            logger.warning(f"Model not found for {tf}, retraining...")
            st.warning(f"Model for {tf} not found — retraining now.")
            model, _ = train_baseline_ml(tf_df, target_col='target', save_model=False)
            joblib.dump(model, model_path)
        else:
            logger.info(f"Loading model from {model_path}")
            model = load_model(model_path)

        logger.info(f"Generating predictions for {tf} timeframe")
        df_preds[f"pred_{tf}"] = predict(model, tf_df[tf_cols])

    return df_preds
