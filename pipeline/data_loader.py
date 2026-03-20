# pipeline/data_loader.py

import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger
from utils.feature_utils import reduce_features

logger = get_logger("data_loader")

# ---------------------------------------------------------------------------
# Columns that leak future information and MUST be removed before training.
# 'future_returns' — pre-computed return using future prices.
# 'signal'         — trading signal derived from future data.
# ---------------------------------------------------------------------------
LEAKING_COLUMNS = ['future_returns', 'signal']

# Patterns that suggest a column may contain forward-looking information.
_SUSPICIOUS_PATTERNS = re.compile(r'(future|forward|next|target)', re.IGNORECASE)


def _warn_suspicious_columns(columns):
    """Log a warning for any column whose name matches a leakage-related keyword."""
    for col in columns:
        if _SUSPICIOUS_PATTERNS.search(col):
            logger.warning(
                f"Potentially leaking column detected: '{col}'. "
                "Verify that it does not contain forward-looking information."
            )


def load_data(filepath, target_horizon=1, target_type='classification'):
    """
    Load and preprocess gold dataset.

    Parameters:
    - filepath: path to the dataset CSV
    - target_horizon: number of future steps to define the label
    - target_type: 'classification' or 'regression'

    Returns:
    - DataFrame with features and target column
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.debug(f"Raw data shape: {df.shape}")

    # ------------------------------------------------------------------
    # Drop known leaking columns BEFORE any other processing so they can
    # never influence feature engineering, normalization, or modelling.
    # ------------------------------------------------------------------
    leak_cols_present = [c for c in LEAKING_COLUMNS if c in df.columns]
    if leak_cols_present:
        df = df.drop(columns=leak_cols_present)
        logger.info(f"Dropped leaking columns: {leak_cols_present}")

    # Warn about any remaining columns whose names look suspicious.
    _warn_suspicious_columns(df.columns)

    # ------------------------------------------------------------------
    # Feature reduction: drop weak CDL_ columns and highly correlated
    # features before target creation and normalization.
    # ------------------------------------------------------------------
    df = reduce_features(df)

    # Sort by date if applicable
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

    # Create target
    if target_type == 'classification':
        df['target'] = (df['Close'].shift(-target_horizon) > df['Close']).astype(int)
    elif target_type == 'regression':
        df['target'] = df['Close'].shift(-target_horizon) / df['Close'] - 1
    else:
        raise ValueError("Invalid target_type. Choose 'classification' or 'regression'.")

    logger.debug("Target column created")

    # Drop rows with NaNs
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Data cleaned and ready. Final shape: {df.shape}")

    return df


def normalize_features(df, exclude_cols=None, train_end_idx=None):
    """
    Normalize numerical features. Fits scaler ONLY on training data to prevent leakage.

    Parameters:
    - df: DataFrame
    - exclude_cols: columns to exclude from normalization (default: target, date, Date)
    - train_end_idx: index marking end of training data. If None, uses 70% of data.

    Returns:
    - DataFrame with normalized features
    - Fitted scaler (fitted only on training portion)
    """
    if exclude_cols is None:
        exclude_cols = ['target', 'date', 'Date']

    logger.info("Normalizing features")

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()

    if train_end_idx is None:
        train_end_idx = int(len(df) * 0.7)

    # Fit scaler ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_end_idx][numeric_cols])

    # Transform ALL data using training-fitted scaler
    df = df.copy()
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    logger.debug(f"Normalized {len(numeric_cols)} columns, scaler fitted on first {train_end_idx} rows")
    return df, scaler


def save_scaler(scaler, path):
    """Save fitted scaler to disk using joblib."""
    import joblib
    joblib.dump(scaler, path)
    logger.info(f"Scaler saved to {path}")


def load_scaler(path):
    """Load fitted scaler from disk."""
    import joblib
    scaler = joblib.load(path)
    logger.info(f"Scaler loaded from {path}")
    return scaler
