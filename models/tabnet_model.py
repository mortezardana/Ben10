# models/tabnet_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
from utils.logger import get_logger
import torch

logger = get_logger("tabnet")


def train_tabnet_model(df, target_col='target', device_name='auto'):
    logger.info("Preparing data for TabNet")

    feature_cols = df.drop(columns=['Date', target_col], errors='ignore').select_dtypes(
        include='number').columns.tolist()
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    clf = TabNetClassifier(device_name=device_name, verbose=0, seed=42)

    logger.info(f"Training TabNet model on {device_name.upper()}")
    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        eval_name=['valid'],
        eval_metric=['accuracy'],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128
    )

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info(f"TabNet Accuracy: {acc:.4f}")

    return clf, {"accuracy": acc}


def predict_tabnet(model, df):
    feature_cols = df.drop(columns=['Date', 'target'], errors='ignore').select_dtypes(include='number').columns.tolist()
    X = df[feature_cols].values
    preds = model.predict(X)
    df_out = df.copy()
    df_out['prediction'] = preds
    return df_out
