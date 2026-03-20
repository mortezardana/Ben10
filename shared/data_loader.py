"""Centralized data loading — the ONE place all data loading happens."""

import pandas as pd
from pathlib import Path
from utils.logger import get_logger
from utils.feature_utils import reduce_features

logger = get_logger("shared.data_loader")


def load_gold_data(config=None):
    """
    Load gold data with all fixes applied.

    Parameters:
    - config: AppConfig or dict. If None, uses defaults.

    Returns:
    - dict with keys: 'train', 'val', 'test', 'scaler', 'full_df'
    """
    # Defaults
    data_path = "Data/gold_4h.csv"
    test_size = 0.2
    val_size = 0.1
    exclude_cols = ['target', 'date', 'Date', 'future_returns', 'signal']

    if config is not None:
        if hasattr(config, 'data'):
            data_path = str(config.data.data_path)
            test_size = config.data.test_size
            val_size = config.data.val_size
            exclude_cols = config.data.exclude_cols
        elif isinstance(config, dict):
            data_path = config.get('data_path', data_path)
            test_size = config.get('test_size', test_size)
            val_size = config.get('val_size', val_size)

    # Load CSV
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")

    # Drop leaking columns
    leaking = ['future_returns', 'signal']
    leak_found = [c for c in leaking if c in df.columns]
    if leak_found:
        df = df.drop(columns=leak_found)
        logger.info(f"Dropped leaking columns: {leak_found}")

    # Handle date column
    date_col = None
    if 'Date' in df.columns:
        date_col = 'Date'
    elif 'date' in df.columns:
        date_col = 'date'

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

    # Feature reduction
    df = reduce_features(df)

    # Create target
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna().reset_index(drop=True)

    # Chronological split
    n = len(df)
    test_n = int(n * test_size)
    val_n = int(n * val_size)
    train_n = n - test_n - val_n

    train_df = df.iloc[:train_n].copy()
    val_df = df.iloc[train_n:train_n + val_n].copy()
    test_df = df.iloc[train_n + val_n:].copy()

    # Normalize — fit ONLY on training data
    from sklearn.preprocessing import StandardScaler
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'scaler': scaler,
        'full_df': df,
        'feature_cols': feature_cols,
    }
