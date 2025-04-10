# utils/multi_timeframe_utils.py

import pandas as pd
from utils.logger import get_logger

logger = get_logger("multi_timeframe_utils")


def load_and_prepare(filepath, timeframe_label, datetime_col='Date'):
    df = pd.read_csv(filepath, parse_dates=[datetime_col])
    df = df.sort_values(datetime_col)
    df = df.set_index(datetime_col)
    df = df.add_suffix(f"_{timeframe_label}")
    df.index.name = datetime_col
    return df


def merge_multi_timeframes(files_by_timeframe, how='outer'):
    """
    files_by_timeframe: dict like {
        '1h': 'data/gold_1h.csv',
        '4h': 'data/gold_4h.csv',
        '1d': 'data/gold_1d.csv'
    }
    Returns merged DataFrame with suffixes per timeframe
    """
    logger.info("Merging multi-timeframe datasets")
    merged_df = None

    for label, path in files_by_timeframe.items():
        logger.info(f"Loading {label} data from {path}")
        df = load_and_prepare(path, label)
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.join(df, how=how)

    merged_df.dropna(inplace=True)
    merged_df.reset_index(inplace=True)
    logger.info(f"Final merged shape: {merged_df.shape}")
    return merged_df
