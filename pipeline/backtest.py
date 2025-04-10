# pipeline/backtest.py

import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger("backtest")


def simple_strategy_backtest(df, entry_col='prediction', price_col='Close'):
    logger.info("Running simple strategy backtest")
    df = df.copy()

    df['position'] = df[entry_col].shift(1).fillna(0)
    df['log_return'] = (df[price_col] / df[price_col].shift(1)).apply(lambda x: np.nan if x <= 0 else np.log(x))
    df['strategy_return'] = df['position'] * df['log_return']
    df['equity_curve'] = df['strategy_return'].cumsum().apply(np.exp)

    # Retain columns needed for metrics and plotting
    keep_cols = ['target', 'prediction', 'position', 'log_return', 'strategy_return', 'equity_curve']
    keep_cols = [col for col in keep_cols if col in df.columns]  # filter missing

    logger.info(f"Total return: {df['equity_curve'].iloc[-1] - 1:.2%}")
    return df[keep_cols].dropna()
