import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("fracdiff")


def _get_weights(d, threshold=1e-5):
    """Compute fractional differentiation weights using the binomial series."""
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    return np.array(weights[::-1])


def frac_diff(series, d=0.2, threshold=1e-5):
    """
    Apply fractional differentiation of order d to a time series.

    d=0: original series (full memory, non-stationary)
    d=1: standard returns (stationary, no memory)
    d~0.2: sweet spot -- stationary with >90% memory retained

    Parameters:
    - series: pd.Series
    - d: differentiation order
    - threshold: minimum weight to include in kernel (truncation)

    Returns:
    - pd.Series of fractionally differenced values
    """
    if d == 0:
        return series.copy()

    weights = _get_weights(d, threshold)
    width = len(weights)

    result = pd.Series(index=series.index, dtype=float)
    values = series.values

    for i in range(width - 1, len(values)):
        window = values[i - width + 1:i + 1]
        result.iloc[i] = np.dot(weights, window)

    return result


def find_min_d(series, p_value=0.05, max_d=1.0, step=0.05):
    """
    Find minimum d that makes the series stationary (ADF test p-value < threshold).

    Returns:
    - float: minimum d for stationarity
    """
    from statsmodels.tsa.stattools import adfuller

    for d in np.arange(0, max_d + step, step):
        diff_series = frac_diff(series, d=d).dropna()
        if len(diff_series) < 20:
            continue
        try:
            adf_result = adfuller(diff_series, maxlag=1)
            if adf_result[1] < p_value:
                logger.info(f"Minimum d for stationarity: {d:.2f} (ADF p-value: {adf_result[1]:.4f})")
                return round(d, 2)
        except Exception:
            continue

    logger.warning("Could not find d for stationarity, returning 1.0")
    return 1.0


def apply_fracdiff_to_prices(df, d=0.2, price_cols=None):
    """
    Apply fractional differentiation to price-level columns.
    Does NOT apply to already-stationary features (RSI, returns, oscillators).

    Parameters:
    - df: DataFrame
    - d: differentiation order
    - price_cols: list of column names to fracdiff. If None, auto-detects.

    Returns:
    - DataFrame with price columns replaced by fracdiff'd versions
    """
    if price_cols is None:
        # Auto-detect price-level columns
        price_patterns = ['close', 'high', 'low', 'open', 'sma', 'ema', 'ma',
                         'bb_upper', 'bb_middle', 'bb_lower', 'kama', 'dema', 'tema',
                         't3', 'trima', 'wma', 'linearreg', 'tsf', 'ht_trendline',
                         'mama', 'fama', 'avgprice', 'medprice', 'typprice', 'wclprice',
                         'midpoint', 'sar', 'sarext']
        price_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(p == col_lower for p in price_patterns):
                price_cols.append(col)

    df = df.copy()
    for col in price_cols:
        if col in df.columns:
            df[col] = frac_diff(df[col], d=d)
            logger.debug(f"Applied fracdiff(d={d}) to {col}")

    logger.info(f"Fractionally differenced {len(price_cols)} price columns with d={d}")
    return df
