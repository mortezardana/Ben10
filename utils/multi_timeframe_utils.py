"""Multi-timeframe data alignment and feature extraction.

Uses merge_asof with backward direction and 1-period shift on higher timeframes
to prevent look-ahead bias.
"""
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger("multi_timeframe")


def align_timeframes(df_4h, df_daily=None, df_weekly=None, df_1h=None):
    """
    Merge multi-timeframe data using merge_asof with backward direction.
    Higher-TF features are SHIFTED by 1 period before merge to ensure
    a 4H bar only sees COMPLETED higher-TF bars.
    """
    result = df_4h.copy()

    if not isinstance(result.index, pd.DatetimeIndex):
        if 'Date' in result.columns:
            result = result.set_index('Date')
        elif 'date' in result.columns:
            result = result.set_index('date')
        result.index = pd.to_datetime(result.index)

    result = result.sort_index()

    if df_daily is not None:
        daily = df_daily.copy()
        if not isinstance(daily.index, pd.DatetimeIndex):
            daily.index = pd.to_datetime(daily.index)
        daily = daily.sort_index()
        # Shift by 1 period — 4H bar should NOT see same-day daily close
        daily = daily.shift(1)
        daily.columns = [f'daily_{c}' if not c.startswith('daily_') else c for c in daily.columns]
        result = pd.merge_asof(result, daily, left_index=True, right_index=True, direction='backward')

    if df_weekly is not None:
        weekly = df_weekly.copy()
        if not isinstance(weekly.index, pd.DatetimeIndex):
            weekly.index = pd.to_datetime(weekly.index)
        weekly = weekly.sort_index()
        weekly = weekly.shift(1)
        weekly.columns = [f'weekly_{c}' if not c.startswith('weekly_') else c for c in weekly.columns]
        result = pd.merge_asof(result, weekly, left_index=True, right_index=True, direction='backward')

    if df_1h is not None:
        agg_1h = aggregate_1h_to_4h(df_1h)
        agg_1h.columns = [f'1h_{c}' if not c.startswith('1h_') else c for c in agg_1h.columns]
        result = pd.merge_asof(result, agg_1h, left_index=True, right_index=True, direction='backward')

    logger.info(f"Aligned timeframes: {result.shape[1]} columns, {len(result)} rows")
    return result


def aggregate_1h_to_4h(df_1h):
    """Compute 1H-derived features aggregated per 4H bar window."""
    df = df_1h.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    agg = df.resample('4h').agg({
        'Close': ['std', 'last', 'first'],
        'Volume': 'sum',
    })
    agg.columns = ['intra_bar_volatility', 'close_last', 'close_first', 'volume_sum']
    agg['intra_bar_trend'] = (agg['close_last'] - agg['close_first']) / agg['close_first'].replace(0, np.nan)
    agg = agg.drop(columns=['close_last', 'close_first'])
    return agg


def compute_daily_features(df_daily):
    """6 features from daily data."""
    df = df_daily.copy()
    close = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
    features = pd.DataFrame(index=df.index)
    features['daily_trend'] = close.pct_change(5)
    features['daily_atr'] = (df['High'] - df['Low']).rolling(14).mean() if 'High' in df.columns else np.nan
    features['daily_rsi'] = _compute_rsi(close, 14)
    sma_20 = close.rolling(20).mean()
    features['daily_ma_distance'] = (close - sma_20) / sma_20.replace(0, np.nan)
    if 'Open' in df.columns:
        features['daily_close_vs_open'] = (close - df['Open']) / df['Open'].replace(0, np.nan)
    if 'High' in df.columns and 'Low' in df.columns:
        rng = df['High'] - df['Low']
        features['daily_range_position'] = (close - df['Low']) / rng.replace(0, np.nan)
    return features


def compute_weekly_features(df_weekly):
    """3 features from weekly data."""
    df = df_weekly.copy()
    close = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
    features = pd.DataFrame(index=df.index)
    features['weekly_trend'] = close.pct_change(4)
    sma_10 = close.rolling(10).mean()
    features['weekly_ma_position'] = (close - sma_10) / sma_10.replace(0, np.nan)
    if 'High' in df.columns:
        features['weekly_atr'] = (df['High'] - df['Low']).rolling(14).mean()
    return features


def _compute_rsi(series, period=14):
    """Simple RSI computation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
