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

    # Normalization
    'exclude_cols': ['target', 'date'],

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
