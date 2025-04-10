# baseline_ml.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from utils.logger import get_logger

logger = get_logger("baseline_ml")


def train_baseline_ml(df, target_col='target', test_size=0.2, val_size=0.1, random_state=42, save_model=True):
    """
    Train a baseline XGBoost classifier on the gold dataset.

    Parameters:
    - df: DataFrame containing features and target
    - target_col: Name of the target column
    - test_size: Proportion for the test split
    - val_size: Proportion for the validation split (from train)
    - random_state: For reproducibility
    - save_model: Whether to save the trained model to disk

    Returns:
    - model: Trained XGBoost model
    - metrics: Dictionary of evaluation metrics
    """

    logger.info("Starting baseline ML training pipeline...")

    # Drop rows with NaNs (e.g., from indicators)
    logger.debug(f"Initial shape: {df.shape}")
    df = df.dropna()
    logger.debug(f"Shape after dropping NaNs: {df.shape}")

    # Split features and target
    X = df.drop(columns=[target_col])

    # Drop non-numeric columns (e.g., 'date')
    X = X.select_dtypes(include=['number'])
    y = df[target_col]

    # Train/val/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size / (1 - test_size), shuffle=False
    )

    logger.info("Data split complete")
    logger.debug(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Initialize and train model
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=random_state
    )

    logger.info("Training XGBoost model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    logger.info("Model training complete")

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.debug(f"Classification Report: {report}")

    metrics = {
        'accuracy': acc,
        'report': report
    }

    # Optionally save model
    if save_model:
        joblib.dump(model, 'models/xgb_baseline.pkl')
        logger.info("Model saved to models/xgb_baseline.pkl")

    return model, metrics
