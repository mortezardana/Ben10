# pipeline/config.py

from pathlib import Path

CONFIG = {
    # Paths
    'data_path': Path("data/gold_4h.csv"),
    'model_output_dir': Path("models/"),
    'log_dir': Path("logs/"),

    # Target configuration
    'target_type': 'classification',  # or 'regression'
    'target_horizon': 1,  # predict 1 candle ahead

    # Model training
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
    'save_model': True,

    # Columns to exclude from features.
    # 'future_returns' and 'signal' are derived from future price data and
    # leak the target variable — they must be dropped before any processing.
    'exclude_cols': ['target', 'date', 'future_returns', 'signal'],

    # Labeling
    'labeling': {
        'method': 'triple_barrier',  # or 'binary'
        'tp_multiplier': 2.0,
        'sl_multiplier': 1.0,
        'max_holding_period': 12,
    },

    # Model hyperparameters
    'xgb_params': {
        'n_estimators': 300,
        'learning_rate': 0.03,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }
}
