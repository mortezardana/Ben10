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
    Keeps the one with higher variance. Never drops OHLCV columns.
    """
    if exclude_cols is None:
        exclude_cols = ['target', 'date', 'Date']

    # Protect core OHLCV columns — they're needed for target creation and backtesting
    protected = {'Open', 'High', 'Low', 'Close', 'Volume'}
    exclude_cols = list(set(exclude_cols) | protected)

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


def drop_redundant_features(df):
    """
    Drop features that are redundant, near-constant, or questionable.

    Keeps the most informative representative from each group:
      Momentum:    roc, pct_change          (drop mom, pct_change2, pct_change5)
      Volatility:  atr, natr                (drop stddev, var, volatility, volatility2, trange)
      Stochastic:  stoch_d, stochrsi_k      (drop stochf_d, stochf_k, stochrsi_d)
      Directional: adx, plus_di, minus_di   (drop adxr, dx, plus_dm, minus_dm)
      Hilbert:     ht_dcperiod, ht_sine     (drop ht_dcphase, ht_inphase, ht_quadrature, ht_leadsine, ht_trendmode)
      Other:       drop sum, sarext, corr, minmaxindex_min, ad, obv
    """
    redundant = [
        # Redundant momentum — roc and pct_change are enough
        'mom', 'pct_change2', 'pct_change5',
        # Redundant volatility — atr and natr capture this
        'stddev', 'var', 'volatility', 'volatility2', 'trange',
        # Redundant stochastic — stoch_d and stochrsi_k are sufficient
        'stochf_d', 'stochf_k', 'stochrsi_d',
        # Redundant directional — adx, plus_di, minus_di cover it
        'adxr', 'dx', 'plus_dm', 'minus_dm',
        # Noisy Hilbert Transform features — keep ht_dcperiod and ht_sine
        'ht_dcphase', 'ht_inphase', 'ht_quadrature', 'ht_leadsine', 'ht_trendmode',
        # Near-constant or index-based
        'minmaxindex_min',
        # Non-stationary cumulative (need differencing to be useful, raw values hurt)
        'ad', 'obv',
        # Low-information or scale-dependent
        'sum', 'sarext', 'corr',
    ]

    to_drop = [c for c in redundant if c in df.columns]
    if to_drop:
        logger.info(f"Dropping {len(to_drop)} redundant features: {sorted(to_drop)}")
        df = df.drop(columns=to_drop)
    return df


def reduce_features(df, corr_threshold=0.95):
    """Apply full feature reduction pipeline: CDL -> redundant -> correlated."""
    original_count = len(df.columns)
    df = drop_cdl_columns(df)
    df = drop_redundant_features(df)
    df = drop_correlated_features(df, threshold=corr_threshold)
    final_count = len(df.columns)
    logger.info(f"Feature reduction: {original_count} -> {final_count} columns")
    return df
