import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("sampling")


def cusum_filter(close, threshold=None):
    """
    Symmetric CUSUM filter. Detects structural breaks in price.

    Parameters:
    - close: pd.Series of close prices
    - threshold: if None, use rolling standard deviation of returns (adaptive)

    Returns:
    - pd.DatetimeIndex of event timestamps where CUSUM triggered
    """
    returns = close.pct_change().dropna()

    if threshold is None:
        # Use daily volatility (std of returns) as adaptive threshold
        threshold = returns.std()

    events = []
    s_pos = 0.0
    s_neg = 0.0

    for i in range(len(returns)):
        r = returns.iloc[i]
        s_pos = max(0, s_pos + r)
        s_neg = min(0, s_neg + r)

        if s_pos > threshold:
            events.append(returns.index[i])
            s_pos = 0.0  # Reset
        elif s_neg < -threshold:
            events.append(returns.index[i])
            s_neg = 0.0  # Reset

    logger.info(f"CUSUM filter: {len(events)} events from {len(close)} bars "
                f"({100 * len(events) / len(close):.1f}%), threshold={threshold:.6f}")

    return pd.DatetimeIndex(events)


def get_cusum_events(df, close_col='Close', threshold=None):
    """
    Return subset of DataFrame rows at CUSUM event points.

    Parameters:
    - df: DataFrame with datetime index or close_col
    - close_col: name of close price column
    - threshold: CUSUM threshold (None = adaptive)

    Returns:
    - DataFrame subset at CUSUM event timestamps
    """
    close = df[close_col]
    event_idx = cusum_filter(close, threshold)

    # Filter df to only event rows
    events_df = df[df.index.isin(event_idx)].copy()
    logger.info(f"CUSUM events: kept {len(events_df)} / {len(df)} rows")
    return events_df
