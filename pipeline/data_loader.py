# pipeline/data_loader.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger

logger = get_logger("data_loader")


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


def normalize_features(df, exclude_cols=['target', 'date']):
    """
    Normalize numerical features.

    Parameters:
    - df: DataFrame
    - exclude_cols: columns to exclude from normalization

    Returns:
    - DataFrame with normalized features
    - Fitted scaler
    """
    logger.info("Normalizing features")

    # Exclude manually listed columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Only keep numeric columns from selected ones
    numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    logger.debug(f"Normalized columns: {numeric_cols}")
    logger.debug("Feature normalization complete")
    return df, scaler
