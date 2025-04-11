# models/lstm_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from utils.logger import get_logger
import tensorflow as tf

logger = get_logger("lstm")


def train_lstm_model(df, device=None):
    """
    Trains an LSTM classification model using TensorFlow.

    Parameters:
    - df: DataFrame containing training data
    - device: Optional device string ('cuda', 'cpu', etc.)

    Returns:
    - model: Trained LSTM model
    - metrics: Dictionary of final loss and accuracy
    """

    # Map device to TensorFlow format
    if device and "cuda" in device.lower():
        tf_device = "/GPU:0"
    else:
        tf_device = "/CPU:0"

    print(f"[INFO] Training LSTM model on {'GPU' if 'GPU' in tf_device else 'CPU'}")

    with tf.device(tf_device):
        # Feature selection
        feature_cols = [col for col in df.columns if col not in ['Date', 'target', 'prediction']]
        X = df[feature_cols].values.reshape((df.shape[0], len(feature_cols), 1))
        y = df['target'].values

        # Build classification model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(len(feature_cols), 1)),
            tf.keras.layers.Dense(1, activation='sigmoid')  # sigmoid for binary classification
        ])

        # Compile with accuracy metric
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Fit model
        history = model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    metrics = {
        "loss": history.history['loss'][-1],
        "accuracy": history.history['accuracy'][-1]
    }

    return model, metrics


def predict_lstm(model, df, device=None):
    feature_cols = df.drop(columns=['Date', 'target'], errors='ignore').select_dtypes(include='number').columns.tolist()
    X = df[feature_cols].values
    X = X.reshape((X.shape[0], X.shape[1], 1))
    with tf.device(f"/{device}:0" if device else "/CPU:0"):
        preds = (model.predict(X) > 0.5).astype(int).flatten()
    df_out = df.copy()
    df_out['prediction'] = preds
    return df_out
