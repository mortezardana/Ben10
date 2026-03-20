# utils/sequence_utils.py

import numpy as np


def create_sequences(data, target, sequence_length=32):
    """
    Create sliding window sequences for time series models.

    Parameters:
    - data: numpy array of shape (n_samples, n_features)
    - target: numpy array of shape (n_samples,)
    - sequence_length: number of timesteps per sequence (default 32 = ~5 trading days at 4H)

    Returns:
    - X: array of shape (n_sequences, sequence_length, n_features)
    - y: array of shape (n_sequences,)
    """
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(target[i])
    return np.array(X), np.array(y)
