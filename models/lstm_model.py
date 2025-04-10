# models/lstm_model.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report
from utils.logger import get_logger

logger = get_logger("lstm_model")


def create_sequences(data, target, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def train_lstm_model(df, target_col='target', sequence_length=32, test_size=0.2, val_size=0.1, epochs=50,
                     batch_size=64):
    logger.info("Preparing data for LSTM training")

    df = df.dropna().reset_index(drop=True)
    X_all = df.drop(columns=[target_col, 'date'], errors='ignore').select_dtypes(include='number')
    y_all = df[target_col]

    X_seq, y_seq = create_sequences(X_all.values, y_all.values, sequence_length)
    logger.debug(f"Created sequences: {X_seq.shape}, Labels: {y_seq.shape}")

    total_samples = len(X_seq)
    test_size = int(test_size * total_samples)
    val_size = int(val_size * total_samples)

    X_train = X_seq[:-val_size - test_size]
    y_train = y_seq[:-val_size - test_size]
    X_val = X_seq[-val_size - test_size:-test_size]
    y_val = y_seq[-val_size - test_size:-test_size]
    X_test = X_seq[-test_size:]
    y_test = y_seq[-test_size:]

    logger.info("Data split complete for LSTM")

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    logger.info("Training LSTM model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    logger.info("Evaluating LSTM model")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        'accuracy': acc,
        'report': report
    }

    logger.info(f"LSTM Test Accuracy: {acc:.4f}")
    return model, metrics


def predict_lstm(model, df, target_col='target', sequence_length=32):
    logger.info("Generating predictions with trained LSTM model")
    df = df.dropna().reset_index(drop=True)

    X_all = df.drop(columns=[target_col, 'date'], errors='ignore').select_dtypes(include='number')
    y_all = df[target_col].values

    X_seq, y_seq = create_sequences(X_all.values, y_all, sequence_length)
    y_pred = (model.predict(X_seq) > 0.5).astype(int).flatten()

    pred_df = df.iloc[sequence_length:].copy()
    pred_df['prediction'] = y_pred

    logger.info(f"Predictions generated for {len(pred_df)} samples")
    return pred_df
