"""Loaders for macro data: DXY, Treasury yields, VIX."""
import pandas as pd
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("data_sources.macro")
CACHE_DIR = Path("data/external/")


def _cache_path(name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.parquet"


def load_dxy(start=None, end=None):
    """Load US Dollar Index via yfinance."""
    cache = _cache_path("dxy")
    if cache.exists():
        df = pd.read_parquet(cache)
        if start: df = df[df.index >= start]
        if end: df = df[df.index <= end]
        return df
    try:
        import yfinance as yf
        dx = yf.download("DX-Y.NYB", start=start, end=end, progress=False)
        dx = dx[['Close']].rename(columns={'Close': 'dxy_close'})
        dx.to_parquet(cache)
        logger.info(f"Downloaded DXY: {len(dx)} rows")
        return dx
    except Exception as e:
        logger.warning(f"Failed to load DXY: {e}")
        return pd.DataFrame()


def load_vix(start=None, end=None):
    """Load VIX via yfinance."""
    cache = _cache_path("vix")
    if cache.exists():
        df = pd.read_parquet(cache)
        if start: df = df[df.index >= start]
        if end: df = df[df.index <= end]
        return df
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start=start, end=end, progress=False)
        vix = vix[['Close']].rename(columns={'Close': 'vix_close'})
        vix.to_parquet(cache)
        logger.info(f"Downloaded VIX: {len(vix)} rows")
        return vix
    except Exception as e:
        logger.warning(f"Failed to load VIX: {e}")
        return pd.DataFrame()


def load_treasury_yields(start=None, end=None):
    """Load 2Y and 10Y Treasury yields via FRED/yfinance."""
    cache = _cache_path("treasury")
    if cache.exists():
        df = pd.read_parquet(cache)
        if start: df = df[df.index >= start]
        if end: df = df[df.index <= end]
        return df
    try:
        import yfinance as yf
        t2y = yf.download("^IRX", start=start, end=end, progress=False)[['Close']].rename(columns={'Close': 'yield_2y'})
        t10y = yf.download("^TNX", start=start, end=end, progress=False)[['Close']].rename(columns={'Close': 'yield_10y'})
        df = t2y.join(t10y, how='outer').ffill()
        df['yield_spread'] = df['yield_10y'] - df['yield_2y']
        df.to_parquet(cache)
        logger.info(f"Downloaded Treasury yields: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"Failed to load Treasury yields: {e}")
        return pd.DataFrame()
