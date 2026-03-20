# models/cnn_lstm.py

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from utils.logger import get_logger
from utils.sequence_utils import create_sequences
import tensorflow as tf

logger = get_logger("cnn_lstm")

SEQUENCE_LENGTH = 32


def train_cnn_lstm_model(df, target_col='target', device=None):
    """
    Trains a CNN-LSTM classification model with proper sliding-window sequence
    inputs of shape (samples, timesteps, features).
    """
    logger.info("Preparing data for CNN-LSTM")

    feature_cols = df.drop(columns=['Date', target_col, 'prediction'], errors='ignore') \
        .select_dtypes(include='number').columns.tolist()
    X_raw = df[feature_cols].values
    y_raw = df[target_col].values

    # Create sliding window sequences: (n_sequences, sequence_length, n_features)
    X_seq, y_seq = create_sequences(X_raw, y_raw, sequence_length=SEQUENCE_LENGTH)
    logger.info(f"Created sequences: X={X_seq.shape}, y={y_seq.shape}")

    # Chronological train / val / test split
    total = len(X_seq)
    test_size = int(0.2 * total)
    val_size = int(0.1 * total)

    X_train = X_seq[:total - val_size - test_size]
    y_train = y_seq[:total - val_size - test_size]
    X_val = X_seq[total - val_size - test_size:total - test_size]
    y_val = y_seq[total - val_size - test_size:total - test_size]
    X_test = X_seq[total - test_size:]
    y_test = y_seq[total - test_size:]

    logger.info(f"Training CNN-LSTM model on {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
    if device and "cuda" in device.lower():
        tf_device = "/GPU:0"
    else:
        tf_device = "/CPU:0"

    with tf.device(tf_device):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                         input_shape=(SEQUENCE_LENGTH, len(feature_cols))))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=64,
            callbacks=[es],
            verbose=0
        )

        preds = (model.predict(X_test) > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)

    logger.info(f"CNN-LSTM Accuracy: {acc:.4f}")
    return model, {"accuracy": acc}


def predict_cnn_lstm(model, df, target_col='target', device=None):
    """
    Generate predictions using a trained CNN-LSTM model with proper windowing.

    Returns a DataFrame (rows aligned to df.iloc[SEQUENCE_LENGTH:]) with a
    'prediction' column.
    """
    feature_cols = df.drop(columns=['Date', 'target', 'prediction'], errors='ignore') \
        .select_dtypes(include='number').columns.tolist()
    X_raw = df[feature_cols].values
    y_raw = df[target_col].values if target_col in df.columns else np.zeros(len(df))

    X_seq, y_seq = create_sequences(X_raw, y_raw, sequence_length=SEQUENCE_LENGTH)

    if device and "cuda" in device.lower():
        tf_device = "/GPU:0"
    else:
        tf_device = "/CPU:0"

    with tf.device(tf_device):
        preds = (model.predict(X_seq) > 0.5).astype(int).flatten()

    df_out = df.iloc[SEQUENCE_LENGTH:].copy()
    df_out['prediction'] = preds
    logger.info(f"Predictions generated for {len(df_out)} samples")
    return df_out
