"""
ICT (Inner Circle Trader) feature engineering.

Computes features based on smart money concepts:
- Fair Value Gaps (FVG)
- Order Blocks (OB)
- Liquidity levels and sweeps
- Market structure (BOS/MSS)
- Killzones (session timing)
- Displacement detection
"""

import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("ict_features")


# ---------------------------------------------------------------------------
# Fair Value Gaps (FVG)
# ---------------------------------------------------------------------------

def detect_fvg(high, low, close, open_):
    """
    Detect Fair Value Gaps — 3-candle imbalances where price moved so fast
    it left a gap between candle 1 and candle 3.

    Bullish FVG: high[i-2] < low[i]  (gap up — expect price to retrace down to fill)
    Bearish FVG: low[i-2] > high[i]  (gap down — expect price to retrace up to fill)

    Returns DataFrame with: fvg_bullish, fvg_bearish, fvg_size, fvg_midpoint
    """
    n = len(high)
    fvg_bullish = np.zeros(n, dtype=int)
    fvg_bearish = np.zeros(n, dtype=int)
    fvg_size = np.zeros(n)
    fvg_mid = np.full(n, np.nan)

    for i in range(2, n):
        # Bullish FVG: candle 1 high < candle 3 low
        if high.iloc[i - 2] < low.iloc[i]:
            fvg_bullish[i] = 1
            fvg_size[i] = low.iloc[i] - high.iloc[i - 2]
            fvg_mid[i] = (low.iloc[i] + high.iloc[i - 2]) / 2

        # Bearish FVG: candle 1 low > candle 3 high
        elif low.iloc[i - 2] > high.iloc[i]:
            fvg_bearish[i] = 1
            fvg_size[i] = low.iloc[i - 2] - high.iloc[i]
            fvg_mid[i] = (low.iloc[i - 2] + high.iloc[i]) / 2

    return pd.DataFrame({
        'fvg_bullish': fvg_bullish,
        'fvg_bearish': fvg_bearish,
        'fvg_size': fvg_size,
    }, index=high.index)


def compute_fvg_features(high, low, close, atr, lookback=50):
    """
    Compute FVG-derived features relative to current price.

    Features:
    - nearest_bull_fvg_dist: distance to nearest unfilled bullish FVG (normalized by ATR)
    - nearest_bear_fvg_dist: distance to nearest unfilled bearish FVG (normalized by ATR)
    - open_bull_fvg_count: number of unfilled bullish FVGs below price
    - open_bear_fvg_count: number of unfilled bearish FVGs above price
    """
    n = len(high)
    nearest_bull_dist = np.full(n, np.nan)
    nearest_bear_dist = np.full(n, np.nan)
    open_bull_count = np.zeros(n, dtype=int)
    open_bear_count = np.zeros(n, dtype=int)

    # Track open FVGs as (top, bottom, direction)
    open_fvgs = []

    for i in range(2, n):
        # Detect new FVGs
        if high.iloc[i - 2] < low.iloc[i]:
            # Bullish FVG: gap between high[i-2] and low[i]
            open_fvgs.append({
                'top': low.iloc[i],
                'bottom': high.iloc[i - 2],
                'direction': 'bull',
                'bar': i,
            })
        elif low.iloc[i - 2] > high.iloc[i]:
            # Bearish FVG
            open_fvgs.append({
                'top': low.iloc[i - 2],
                'bottom': high.iloc[i],
                'direction': 'bear',
                'bar': i,
            })

        # Remove filled FVGs and those older than lookback
        # Only check fill for FVGs from PREVIOUS bars (age >= 1), not the one just created
        current_low = low.iloc[i]
        current_high = high.iloc[i]
        still_open = []
        for fvg in open_fvgs:
            age = i - fvg['bar']
            if age > lookback:
                continue
            if age < 1:
                # Just created this bar — don't check fill yet
                still_open.append(fvg)
                continue
            # Bullish FVG filled when close enters the gap midpoint
            mid = (fvg['top'] + fvg['bottom']) / 2
            if fvg['direction'] == 'bull' and current_low <= mid:
                continue  # Filled
            # Bearish FVG filled when close enters the gap midpoint
            if fvg['direction'] == 'bear' and current_high >= mid:
                continue  # Filled
            still_open.append(fvg)
        open_fvgs = still_open

        # Count and find nearest
        price = close.iloc[i]
        atr_val = atr.iloc[i] if not pd.isna(atr.iloc[i]) and atr.iloc[i] > 0 else 1.0

        bull_dists = []
        bear_dists = []
        for fvg in open_fvgs:
            mid = (fvg['top'] + fvg['bottom']) / 2
            if fvg['direction'] == 'bull':
                open_bull_count[i] += 1
                bull_dists.append(abs(price - mid) / atr_val)
            else:
                open_bear_count[i] += 1
                bear_dists.append(abs(price - mid) / atr_val)

        if bull_dists:
            nearest_bull_dist[i] = min(bull_dists)
        if bear_dists:
            nearest_bear_dist[i] = min(bear_dists)

    return pd.DataFrame({
        'nearest_bull_fvg_dist': nearest_bull_dist,
        'nearest_bear_fvg_dist': nearest_bear_dist,
        'open_bull_fvg_count': open_bull_count,
        'open_bear_fvg_count': open_bear_count,
    }, index=high.index)


# ---------------------------------------------------------------------------
# Order Blocks
# ---------------------------------------------------------------------------

def detect_order_blocks(open_, high, low, close, atr, displacement_mult=1.5):
    """
    Detect order blocks — the last opposing candle before a displacement move.

    Bullish OB: last bearish (red) candle before a bullish displacement (>1.5x ATR)
    Bearish OB: last bullish (green) candle before a bearish displacement

    Features:
    - ob_bullish: 1 at the candle that IS a bullish order block
    - ob_bearish: 1 at the candle that IS a bearish order block
    - nearest_bull_ob_dist: distance to nearest bullish OB (ATR-normalized)
    - nearest_bear_ob_dist: distance to nearest bearish OB (ATR-normalized)
    """
    n = len(close)
    ob_bullish = np.zeros(n, dtype=int)
    ob_bearish = np.zeros(n, dtype=int)
    nearest_bull_ob = np.full(n, np.nan)
    nearest_bear_ob = np.full(n, np.nan)

    # Candle direction
    is_bearish = close < open_
    is_bullish = close > open_

    # Displacement: move > displacement_mult * ATR
    move = close - open_

    bull_obs = []  # list of (price_low, price_high, bar_idx)
    bear_obs = []

    for i in range(1, n):
        atr_val = atr.iloc[i] if not pd.isna(atr.iloc[i]) and atr.iloc[i] > 0 else 1.0

        # Check for bullish displacement at bar i
        if move.iloc[i] > displacement_mult * atr_val:
            # Look back for the last bearish candle
            for j in range(i - 1, max(i - 5, 0) - 1, -1):
                if is_bearish.iloc[j]:
                    ob_bullish[j] = 1
                    bull_obs.append((low.iloc[j], high.iloc[j], j))
                    break

        # Check for bearish displacement
        if move.iloc[i] < -displacement_mult * atr_val:
            for j in range(i - 1, max(i - 5, 0) - 1, -1):
                if is_bullish.iloc[j]:
                    ob_bearish[j] = 1
                    bear_obs.append((low.iloc[j], high.iloc[j], j))
                    break

        # Compute distance to nearest OBs (within last 100 bars)
        price = close.iloc[i]

        # Nearest bullish OB (support zones below price)
        best_bull = np.nan
        for ob_low, ob_high, bar in reversed(bull_obs):
            if i - bar > 100:
                break
            mid = (ob_low + ob_high) / 2
            dist = abs(price - mid) / atr_val
            if np.isnan(best_bull) or dist < best_bull:
                best_bull = dist
        nearest_bull_ob[i] = best_bull

        # Nearest bearish OB (resistance zones above price)
        best_bear = np.nan
        for ob_low, ob_high, bar in reversed(bear_obs):
            if i - bar > 100:
                break
            mid = (ob_low + ob_high) / 2
            dist = abs(price - mid) / atr_val
            if np.isnan(best_bear) or dist < best_bear:
                best_bear = dist
        nearest_bear_ob[i] = best_bear

    return pd.DataFrame({
        'ob_bullish': ob_bullish,
        'ob_bearish': ob_bearish,
        'nearest_bull_ob_dist': nearest_bull_ob,
        'nearest_bear_ob_dist': nearest_bear_ob,
    }, index=close.index)


# ---------------------------------------------------------------------------
# Liquidity Levels & Sweeps
# ---------------------------------------------------------------------------

def compute_liquidity_features(high, low, close, atr, swing_period=10, lookback=50):
    """
    Identify liquidity pools (swing highs/lows) and detect sweeps.

    Features:
    - buy_liq_levels: count of swing highs (buy-side liquidity) above price within lookback
    - sell_liq_levels: count of swing lows (sell-side liquidity) below price within lookback
    - dist_nearest_buy_liq: ATR-normalized distance to nearest swing high above
    - dist_nearest_sell_liq: ATR-normalized distance to nearest swing low below
    - liq_sweep: 1 when price pierces a swing high/low then reverses (stop hunt)
    """
    n = len(high)

    # Detect swing highs and lows
    swing_highs = []  # (price, bar_idx)
    swing_lows = []

    for i in range(swing_period, n - swing_period):
        # Swing high: highest high in window
        window_highs = high.iloc[i - swing_period:i + swing_period + 1]
        if high.iloc[i] == window_highs.max():
            swing_highs.append((high.iloc[i], i))

        # Swing low
        window_lows = low.iloc[i - swing_period:i + swing_period + 1]
        if low.iloc[i] == window_lows.min():
            swing_lows.append((low.iloc[i], i))

    # Now compute features per bar
    buy_liq_count = np.zeros(n, dtype=int)
    sell_liq_count = np.zeros(n, dtype=int)
    dist_buy_liq = np.full(n, np.nan)
    dist_sell_liq = np.full(n, np.nan)
    liq_sweep = np.zeros(n, dtype=int)

    sh_idx = 0  # pointer into swing_highs
    sl_idx = 0  # pointer into swing_lows

    for i in range(swing_period, n):
        price = close.iloc[i]
        atr_val = atr.iloc[i] if not pd.isna(atr.iloc[i]) and atr.iloc[i] > 0 else 1.0

        # Count swing highs above price (buy-side liquidity) within lookback
        buy_count = 0
        nearest_buy = np.nan
        for sh_price, sh_bar in reversed(swing_highs):
            if sh_bar < i - lookback:
                break
            if sh_bar >= i:
                continue
            if sh_price > price:
                buy_count += 1
                dist = (sh_price - price) / atr_val
                if np.isnan(nearest_buy) or dist < nearest_buy:
                    nearest_buy = dist

        # Count swing lows below price (sell-side liquidity)
        sell_count = 0
        nearest_sell = np.nan
        for sl_price, sl_bar in reversed(swing_lows):
            if sl_bar < i - lookback:
                break
            if sl_bar >= i:
                continue
            if sl_price < price:
                sell_count += 1
                dist = (price - sl_price) / atr_val
                if np.isnan(nearest_sell) or dist < nearest_sell:
                    nearest_sell = dist

        buy_liq_count[i] = buy_count
        sell_liq_count[i] = sell_count
        dist_buy_liq[i] = nearest_buy
        dist_sell_liq[i] = nearest_sell

        # Sweep detection: price pierced a swing level then reversed
        if i >= 2:
            # Bullish sweep: low pierced a swing low then close recovered above it
            for sl_price, sl_bar in reversed(swing_lows):
                if sl_bar >= i - 1 or sl_bar < i - lookback:
                    continue
                if low.iloc[i] < sl_price and close.iloc[i] > sl_price:
                    liq_sweep[i] = 1  # bullish sweep (stop hunt below)
                    break

            # Bearish sweep: high pierced a swing high then close fell below
            if liq_sweep[i] == 0:
                for sh_price, sh_bar in reversed(swing_highs):
                    if sh_bar >= i - 1 or sh_bar < i - lookback:
                        continue
                    if high.iloc[i] > sh_price and close.iloc[i] < sh_price:
                        liq_sweep[i] = -1  # bearish sweep (stop hunt above)
                        break

    return pd.DataFrame({
        'buy_liq_levels': buy_liq_count,
        'sell_liq_levels': sell_liq_count,
        'dist_nearest_buy_liq': dist_buy_liq,
        'dist_nearest_sell_liq': dist_sell_liq,
        'liq_sweep': liq_sweep,
    }, index=high.index)


# ---------------------------------------------------------------------------
# Market Structure (Break of Structure / Market Structure Shift)
# ---------------------------------------------------------------------------

def compute_market_structure(high, low, close, swing_period=10):
    """
    Track market structure via swing high/low sequences.

    Features:
    - structure: 1=bullish (HH+HL), -1=bearish (LH+LL), 0=ranging
    - bos: 1=bullish break of structure, -1=bearish BOS, 0=none
    - structure_strength: consecutive bars in current structure direction
    """
    n = len(close)
    structure = np.zeros(n, dtype=int)
    bos = np.zeros(n, dtype=int)
    strength = np.zeros(n, dtype=int)

    # Track recent swing points
    last_swing_high = np.nan
    prev_swing_high = np.nan
    last_swing_low = np.nan
    prev_swing_low = np.nan
    current_structure = 0
    current_strength = 0

    for i in range(swing_period, n - swing_period):
        # Detect swing high
        window_h = high.iloc[i - swing_period:i + swing_period + 1]
        if high.iloc[i] == window_h.max():
            prev_swing_high = last_swing_high
            last_swing_high = high.iloc[i]

        # Detect swing low
        window_l = low.iloc[i - swing_period:i + swing_period + 1]
        if low.iloc[i] == window_l.min():
            prev_swing_low = last_swing_low
            last_swing_low = low.iloc[i]

        # Determine structure
        hh = not np.isnan(prev_swing_high) and last_swing_high > prev_swing_high
        hl = not np.isnan(prev_swing_low) and last_swing_low > prev_swing_low
        lh = not np.isnan(prev_swing_high) and last_swing_high < prev_swing_high
        ll = not np.isnan(prev_swing_low) and last_swing_low < prev_swing_low

        new_structure = 0
        if hh and hl:
            new_structure = 1  # Bullish
        elif lh and ll:
            new_structure = -1  # Bearish

        # Break of structure
        if new_structure != 0 and new_structure != current_structure:
            bos[i] = new_structure
            current_strength = 0

        if new_structure != 0:
            current_structure = new_structure
            current_strength += 1

        structure[i] = current_structure
        strength[i] = current_strength

    # Forward-fill for bars within swing_period of the end
    for i in range(n - swing_period, n):
        structure[i] = structure[n - swing_period - 1] if n > swing_period else 0
        strength[i] = strength[n - swing_period - 1] if n > swing_period else 0

    return pd.DataFrame({
        'market_structure': structure,
        'bos_signal': bos,
        'structure_strength': strength,
    }, index=close.index)


# ---------------------------------------------------------------------------
# Killzones (Session Timing)
# ---------------------------------------------------------------------------

def compute_killzones(dates):
    """
    Compute session/killzone features from timestamps.

    Killzones (EST/NY time):
    - Asian:       20:00 - 00:00
    - London Open: 02:00 - 05:00
    - NY Open:     07:00 - 10:00
    - London Close: 10:00 - 12:00

    Features:
    - killzone: categorical (0=off-hours, 1=asian, 2=london, 3=ny_open, 4=london_close)
    - is_killzone: binary — is this bar in any killzone
    - day_of_week: 0-4 (Mon-Fri)
    - hour: 0-23
    """
    dates = pd.to_datetime(dates)

    hour = dates.dt.hour if hasattr(dates, 'dt') else dates.hour
    dow = dates.dt.dayofweek if hasattr(dates, 'dt') else dates.dayofweek

    killzone = np.zeros(len(dates), dtype=int)

    for i in range(len(dates)):
        h = hour.iloc[i] if hasattr(hour, 'iloc') else hour[i]
        if 20 <= h or h < 0:
            killzone[i] = 1  # Asian
        elif 2 <= h < 5:
            killzone[i] = 2  # London Open
        elif 7 <= h < 10:
            killzone[i] = 3  # NY Open
        elif 10 <= h < 12:
            killzone[i] = 4  # London Close

    return pd.DataFrame({
        'killzone': killzone,
        'is_killzone': (killzone > 0).astype(int),
        'day_of_week': dow.values if hasattr(dow, 'values') else dow,
        'hour': hour.values if hasattr(hour, 'values') else hour,
    }, index=dates)


# ---------------------------------------------------------------------------
# Displacement Detection
# ---------------------------------------------------------------------------

def detect_displacement(open_, close, atr, multiplier=1.5):
    """
    Detect displacement candles — strong institutional moves.

    A displacement candle has |close - open| > multiplier * ATR.

    Features:
    - displacement: 1=bullish displacement, -1=bearish, 0=none
    - displacement_strength: magnitude of move / ATR
    - bars_since_displacement: how many bars since last displacement
    """
    n = len(close)
    displacement = np.zeros(n, dtype=int)
    disp_strength = np.zeros(n)
    bars_since = np.zeros(n, dtype=int)

    move = close.values - open_.values
    last_disp_bar = -999

    for i in range(n):
        atr_val = atr.iloc[i] if not pd.isna(atr.iloc[i]) and atr.iloc[i] > 0 else 1.0

        if abs(move[i]) > multiplier * atr_val:
            displacement[i] = 1 if move[i] > 0 else -1
            disp_strength[i] = abs(move[i]) / atr_val
            last_disp_bar = i

        bars_since[i] = i - last_disp_bar if last_disp_bar >= 0 else 999

    return pd.DataFrame({
        'displacement': displacement,
        'displacement_strength': disp_strength,
        'bars_since_displacement': bars_since,
    }, index=close.index)


# ---------------------------------------------------------------------------
# Aggregate: compute all ICT features
# ---------------------------------------------------------------------------

def compute_all_ict_features(df, atr_col='atr'):
    """
    Compute all ICT features from OHLCV + ATR data.

    Parameters:
    - df: DataFrame with Open, High, Low, Close, Volume, Date columns
    - atr_col: name of ATR column (pre-computed). If missing, computes a simple ATR.

    Returns:
    - DataFrame with all ICT features, same index as input
    """
    open_ = df['Open']
    high = df['High']
    low = df['Low']
    close = df['Close']

    # ATR — use pre-computed if available, otherwise compute
    if atr_col in df.columns:
        atr = df[atr_col]
    else:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

    logger.info("Computing ICT features...")

    # FVG features
    fvg = detect_fvg(high, low, close, open_)
    fvg_advanced = compute_fvg_features(high, low, close, atr)

    # Order blocks
    ob = detect_order_blocks(open_, high, low, close, atr)

    # Liquidity
    liq = compute_liquidity_features(high, low, close, atr)

    # Market structure
    ms = compute_market_structure(high, low, close)

    # Killzones
    date_col = df['Date'] if 'Date' in df.columns else df.index
    kz = compute_killzones(date_col)
    kz.index = df.index

    # Displacement
    disp = detect_displacement(open_, close, atr)

    # Combine all
    result = pd.concat([fvg, fvg_advanced, ob, liq, ms, kz, disp], axis=1)

    logger.info(f"ICT features computed: {len(result.columns)} features")
    return result
