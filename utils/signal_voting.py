# utils/signal_voting.py

import pandas as pd


def vote_signals(df: pd.DataFrame, col_fast: str = 'pred_1h', col_slow: str = 'pred_1d',
                 mode: str = 'confirm') -> pd.DataFrame:
    """
    Combine signals from different timeframes using a basic voting logic.

    Parameters:
    - df: DataFrame with prediction columns (e.g. 'pred_1h', 'pred_1d')
    - col_fast: name of the fast signal column (e.g. 1H predictions)
    - col_slow: name of the slow trend column (e.g. 1D predictions)
    - mode: voting mode, 'confirm' (both must agree), 'filter' (allow fast only if slow is positive), etc.

    Returns:
    - df with an added 'voted_signal' column
    """
    if col_fast not in df.columns or col_slow not in df.columns:
        raise ValueError(f"Missing one of the required columns: '{col_fast}', '{col_slow}'")

    if mode == 'confirm':
        df['voted_signal'] = ((df[col_fast] == 1) & (df[col_slow] == 1)).astype(int)
    elif mode == 'filter':
        df['voted_signal'] = df[col_fast].where(df[col_slow] == 1, 0)
    elif mode == 'majority':
        df['voted_signal'] = ((df[[col_fast, col_slow]].sum(axis=1)) >= 1).astype(int)
    else:
        raise ValueError(f"Unsupported voting mode: {mode}")

    return df


def label_for_voting(row):
    """Optional label for analysis purposes"""
    return {
        (0, 0): 'no_signal',
        (1, 0): 'fast_only',
        (0, 1): 'slow_only',
        (1, 1): 'confirmed'
    }.get((row['pred_1h'], row['pred_1d']), 'unknown')
