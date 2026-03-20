"""COMEX open interest data loader."""
import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("data_sources.comex")
CACHE_DIR = Path("data/external/")


def load_open_interest(start=None, end=None):
    """Load COMEX gold open interest data."""
    cache = CACHE_DIR / "comex_oi.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        if start: df = df[df.index >= start]
        if end: df = df[df.index <= end]
        return df
    logger.warning("COMEX OI data not available. Save to data/external/comex_oi.parquet")
    return pd.DataFrame()


def compute_oi_features(oi_data, prices=None):
    """Compute open interest features."""
    if oi_data.empty:
        return pd.DataFrame()

    features = pd.DataFrame(index=oi_data.index)

    if 'open_interest' in oi_data.columns:
        oi = oi_data['open_interest']
        features['oi_change'] = oi.pct_change()
        features['oi_ma_ratio'] = oi / oi.rolling(20).mean()

        if prices is not None and len(prices) > 0:
            aligned_prices = prices.reindex(oi.index, method='ffill')
            price_up = aligned_prices.diff() > 0
            oi_up = oi.diff() > 0
            # Price up + OI up = trend confirmation
            features['oi_price_confirm'] = ((price_up & oi_up) | (~price_up & ~oi_up)).astype(int)
            features['oi_divergence'] = ((price_up & ~oi_up) | (~price_up & oi_up)).astype(int)

    return features
