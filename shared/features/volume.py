"""Volume-based feature engineering for gold trading."""
import numpy as np
import pandas as pd


def compute_obv(close, volume):
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def compute_mfi(high, low, close, volume, period=14):
    """Money Flow Index (0-100)."""
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = pd.Series(0.0, index=close.index)
    neg_mf = pd.Series(0.0, index=close.index)
    tp_diff = tp.diff()
    pos_mf[tp_diff > 0] = mf[tp_diff > 0]
    neg_mf[tp_diff <= 0] = mf[tp_diff <= 0]
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mr))
    return mfi.fillna(50)

def compute_ad(high, low, close, volume):
    """Accumulation/Distribution line."""
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    clv = clv.fillna(0)
    return (clv * volume).cumsum()

def compute_vwap(high, low, close, volume):
    """Volume Weighted Average Price (rolling 20-period)."""
    tp = (high + low + close) / 3
    cum_tp_vol = (tp * volume).rolling(20).sum()
    cum_vol = volume.rolling(20).sum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)

def compute_obv_slope(obv, period=4):
    """Slope of OBV over period."""
    return obv.diff(period) / period

def detect_obv_divergence(close, obv, period=14):
    """Detect OBV divergence (price up + OBV down or vice versa)."""
    price_change = close.diff(period)
    obv_change = obv.diff(period)
    divergence = ((price_change > 0) & (obv_change < 0)) | ((price_change < 0) & (obv_change > 0))
    return divergence.astype(int)

def compute_ad_slope(ad, period=4):
    """Slope of A/D line."""
    return ad.diff(period) / period

def compute_volume_profile(close, volume, period=20):
    """Compute volume profile features: POC, VA high/low."""
    poc = pd.Series(np.nan, index=close.index)
    va_high = pd.Series(np.nan, index=close.index)
    va_low = pd.Series(np.nan, index=close.index)

    for i in range(period, len(close)):
        window_close = close.iloc[i-period:i]
        window_vol = volume.iloc[i-period:i]
        n_bins = 10
        bins = np.linspace(window_close.min(), window_close.max(), n_bins + 1)
        bin_idx = np.digitize(window_close.values, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        bin_vol = np.zeros(n_bins)
        for j in range(len(bin_idx)):
            bin_vol[bin_idx[j]] += window_vol.iloc[j]
        poc_bin = np.argmax(bin_vol)
        poc.iloc[i] = (bins[poc_bin] + bins[poc_bin + 1]) / 2
        total_vol = bin_vol.sum()
        if total_vol > 0:
            sorted_bins = np.argsort(bin_vol)[::-1]
            cumul = 0
            va_bins = []
            for b in sorted_bins:
                cumul += bin_vol[b]
                va_bins.append(b)
                if cumul >= 0.7 * total_vol:
                    break
            va_low.iloc[i] = bins[min(va_bins)]
            va_high.iloc[i] = bins[max(va_bins) + 1]

    return pd.DataFrame({'poc': poc, 'va_high': va_high, 'va_low': va_low}, index=close.index)

def compute_distance_from_poc(close, poc):
    """Distance from Point of Control."""
    return (close - poc) / poc.replace(0, np.nan)

def compute_all_volume_features(df):
    """Compute all volume features from OHLCV data."""
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
    obv = compute_obv(close, volume)
    ad = compute_ad(high, low, close, volume)
    vp = compute_volume_profile(close, volume)

    features = pd.DataFrame(index=df.index)
    features['obv'] = obv
    features['obv_slope'] = compute_obv_slope(obv)
    features['obv_divergence'] = detect_obv_divergence(close, obv)
    features['mfi'] = compute_mfi(high, low, close, volume)
    features['ad_line'] = ad
    features['ad_slope'] = compute_ad_slope(ad)
    features['vwap'] = compute_vwap(high, low, close, volume)
    features['poc'] = vp['poc']
    features['va_high'] = vp['va_high']
    features['va_low'] = vp['va_low']
    features['dist_from_poc'] = compute_distance_from_poc(close, vp['poc'])
    return features
