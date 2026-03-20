import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger("feature_utils")


def drop_cdl_columns(df):
    """Drop all CDL_ prefixed candlestick pattern columns."""
    cdl_cols = [c for c in df.columns if c.startswith('CDL')]
    if cdl_cols:
        logger.info(f"Dropping {len(cdl_cols)} CDL_ columns")
        df = df.drop(columns=cdl_cols)
    return df


def drop_correlated_features(df, threshold=0.95, exclude_cols=None):
    """
    Drop one of each pair of features with correlation > threshold.
    Keeps the one with higher variance.
    """
    if exclude_cols is None:
        exclude_cols = ['target', 'date', 'Date']

    numeric = df.select_dtypes(include='number')
    numeric = numeric.drop(columns=[c for c in exclude_cols if c in numeric.columns], errors='ignore')

    corr_matrix = numeric.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        for corr_col in high_corr:
            if corr_col in to_drop:
                continue
            # Drop the one with lower variance
            if numeric[col].var() >= numeric[corr_col].var():
                to_drop.add(corr_col)
            else:
                to_drop.add(col)

    if to_drop:
        logger.info(f"Dropping {len(to_drop)} correlated features (threshold={threshold}): {sorted(to_drop)}")
        df = df.drop(columns=list(to_drop))
    return df


def get_feature_report(df, exclude_cols=None):
    """Return a dict with feature count info."""
    if exclude_cols is None:
        exclude_cols = ['target', 'date', 'Date']
    numeric = df.select_dtypes(include='number')
    feature_cols = [c for c in numeric.columns if c not in exclude_cols]
    return {
        'total_features': len(feature_cols),
        'feature_names': feature_cols,
    }


def reduce_features(df, corr_threshold=0.95):
    """Apply full feature reduction pipeline: drop CDL, then correlated features."""
    original_count = len(df.columns)
    df = drop_cdl_columns(df)
    df = drop_correlated_features(df, threshold=corr_threshold)
    final_count = len(df.columns)
    logger.info(f"Feature reduction: {original_count} -> {final_count} columns")
    return df
