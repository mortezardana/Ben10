import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger("labeling")


def triple_barrier_labels(close, high, low, atr, tp_multiplier=2.0, sl_multiplier=1.0, max_holding_period=12):
    """
    Triple barrier labeling (Lopez de Prado).

    For each bar, look forward up to max_holding_period bars:
    - If price hits tp_multiplier * ATR above entry first -> label = 1 (win)
    - If price hits sl_multiplier * ATR below entry first -> label = -1 (loss)
    - If neither within max_holding_period -> label = 0 (flat/timeout)

    Uses HIGH and LOW prices (not just close) for barrier checks.

    Returns DataFrame with columns: label, barrier_hit ('tp', 'sl', 'time'), bars_to_hit
    """
    n = len(close)
    labels = np.zeros(n)
    barrier_hit = [''] * n
    bars_to_hit = np.zeros(n, dtype=int)

    for i in range(n):
        entry = close.iloc[i]
        atr_val = atr.iloc[i]

        if pd.isna(atr_val) or atr_val == 0:
            labels[i] = 0
            barrier_hit[i] = 'time'
            bars_to_hit[i] = 0
            continue

        tp_barrier = entry + tp_multiplier * atr_val
        sl_barrier = entry - sl_multiplier * atr_val

        end_idx = min(i + max_holding_period, n - 1)
        hit = False

        for j in range(i + 1, end_idx + 1):
            # Check if high hits take-profit
            if high.iloc[j] >= tp_barrier:
                labels[i] = 1
                barrier_hit[i] = 'tp'
                bars_to_hit[i] = j - i
                hit = True
                break
            # Check if low hits stop-loss
            if low.iloc[j] <= sl_barrier:
                labels[i] = -1
                barrier_hit[i] = 'sl'
                bars_to_hit[i] = j - i
                hit = True
                break

        if not hit:
            labels[i] = 0
            barrier_hit[i] = 'time'
            bars_to_hit[i] = end_idx - i

    result = pd.DataFrame({
        'label': labels.astype(int),
        'barrier_hit': barrier_hit,
        'bars_to_hit': bars_to_hit,
    }, index=close.index)

    logger.info(f"Triple barrier labels: +1={int((labels==1).sum())}, -1={int((labels==-1).sum())}, 0={int((labels==0).sum())}")
    return result


def binary_labels(close, horizon=1):
    """Simple binary up/down label for backward compatibility."""
    labels = (close.shift(-horizon) > close).astype(int)
    return pd.DataFrame({'label': labels}, index=close.index)
