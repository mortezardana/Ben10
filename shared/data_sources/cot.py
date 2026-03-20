"""COT (Commitment of Traders) data loader for gold."""
import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("data_sources.cot")
CACHE_DIR = Path("data/external/")


def load_cot_gold(start=None, end=None):
    """Load COT data for gold from CFTC.
    Note: This requires manual download or API access.
    Returns empty DataFrame if not available.
    """
    cache = CACHE_DIR / "cot_gold.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        if start: df = df[df.index >= start]
        if end: df = df[df.index <= end]
        return df
    logger.warning("COT data not available. Download from CFTC and save to data/external/cot_gold.parquet")
    return pd.DataFrame()


def compute_cot_features(cot_data):
    """Compute features from COT data."""
    if cot_data.empty:
        return pd.DataFrame()

    features = pd.DataFrame(index=cot_data.index)

    if 'non_commercial_long' in cot_data.columns and 'non_commercial_short' in cot_data.columns:
        net = cot_data['non_commercial_long'] - cot_data['non_commercial_short']
        features['cot_net_speculative'] = net
        features['cot_net_change'] = net.diff()
        # Extreme: z-score > 2 or < -2
        mean_net = net.rolling(52).mean()
        std_net = net.rolling(52).std()
        z = (net - mean_net) / std_net.replace(0, np.nan)
        features['cot_extreme'] = (z.abs() > 2).astype(int)

    if 'commercial_long' in cot_data.columns and 'commercial_short' in cot_data.columns:
        comm_net = cot_data['commercial_long'] - cot_data['commercial_short']
        total = cot_data['commercial_long'] + cot_data['commercial_short']
        features['cot_commercial_hedge_ratio'] = comm_net / total.replace(0, np.nan)

    return features
