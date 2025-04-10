# models/predictor.py

import pandas as pd
import joblib
from utils.logger import get_logger

logger = get_logger("predictor")


def load_model(model_path):
    """
    Load a saved model from disk.

    Parameters:
    - model_path: path to .pkl file

    Returns:
    - loaded model
    """
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    return model


def predict(model, df, exclude_cols=['target', 'date']):
    """
    Make predictions using the loaded model.

    Parameters:
    - model: trained model
    - df: input DataFrame with features
    - exclude_cols: columns to exclude from input

    Returns:
    - predictions as a pandas Series
    """
    logger.info("Preparing data for prediction")

    # Drop non-numeric columns and exclude_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=['number'])

    logger.debug(f"Using {len(X.columns)} features for prediction")

    preds = model.predict(X)
    logger.info(f"Generated {len(preds)} predictions")
    return pd.Series(preds, index=df.index, name='prediction')
