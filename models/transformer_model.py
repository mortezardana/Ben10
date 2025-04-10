# models/transformer_model.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, \
    GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report
from utils.logger import get_logger

logger = get_logger("transformer_model")


def create_sequences(data, target, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def transformer_block(inputs, head_size=64, num_heads=2, ff_dim=128, dropout=0.1):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)

    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dropout(dropout)(x_ff)

    # Ensure dimensions match before residual connection
    x_proj = Dense(ff_dim)(x) if x.shape[-1] != ff_dim else x

    x = LayerNormalization(epsilon=1e-6)(x_proj + x_ff)
    return x


def train_transformer_model(df, target_col='target', sequence_length=32, test_size=0.2, val_size=0.1, epochs=50,
                            batch_size=64):
    logger.info("Preparing data for Transformer training")

    df = df.dropna().reset_index(drop=True)
    X_all = df.drop(columns=[target_col, 'date'], errors='ignore').select_dtypes(include='number')
    y_all = df[target_col]

    X_seq, y_seq = create_sequences(X_all.values, y_all.values, sequence_length)

    total_samples = len(X_seq)
    test_size = int(test_size * total_samples)
    val_size = int(val_size * total_samples)

    X_train = X_seq[:-val_size - test_size]
    y_train = y_seq[:-val_size - test_size]
    X_val = X_seq[-val_size - test_size:-test_size]
    y_val = y_seq[-val_size - test_size:-test_size]
    X_test = X_seq[-test_size:]
    y_test = y_seq[-test_size:]

    logger.info("Building Transformer model")

    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = transformer_block(inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    logger.info("Training Transformer model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    logger.info("Evaluating Transformer model")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        'accuracy': acc,
        'report': report
    }

    logger.info(f"Transformer Test Accuracy: {acc:.4f}")
    return model, metrics


def predict_transformer(model, df, target_col='target', sequence_length=32):
    logger.info("Generating predictions with trained Transformer model")
    df = df.dropna().reset_index(drop=True)

    X_all = df.drop(columns=[target_col, 'date'], errors='ignore').select_dtypes(include='number')
    y_all = df[target_col].values

    X_seq, y_seq = create_sequences(X_all.values, y_all, sequence_length)
    y_pred = (model.predict(X_seq) > 0.5).astype(int).flatten()

    pred_df = df.iloc[sequence_length:].copy()
    pred_df['prediction'] = y_pred

    logger.info(f"Predictions generated for {len(pred_df)} samples")
    return pred_df
