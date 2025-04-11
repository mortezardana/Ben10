# models/cnn_lstm.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from utils.logger import get_logger
import tensorflow as tf

logger = get_logger("cnn_lstm")


def train_cnn_lstm_model(df, target_col='target', device=None):
    logger.info("Preparing data for CNN-LSTM")

    feature_cols = df.drop(columns=['Date', target_col], errors='ignore').select_dtypes(
        include='number').columns.tolist()
    X = df[feature_cols].values
    y = df[target_col].values
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    logger.info(f"Training CNN-LSTM model on {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
    with tf.device(f"/{device}:0" if device else "/CPU:0"):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=X_train.shape[1:]))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[es], verbose=0)

        preds = (model.predict(X_test) > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)

    logger.info(f"CNN-LSTM Accuracy: {acc:.4f}")
    return model, {"accuracy": acc}


def predict_cnn_lstm(model, df, device=None):
    feature_cols = df.drop(columns=['Date', 'target'], errors='ignore').select_dtypes(include='number').columns.tolist()
    X = df[feature_cols].values
    X = X.reshape((X.shape[0], X.shape[1], 1))
    with tf.device(f"/{device}:0" if device else "/CPU:0"):
        preds = (model.predict(X) > 0.5).astype(int).flatten()
    df_out = df.copy()
    df_out['prediction'] = preds
    return df_out
