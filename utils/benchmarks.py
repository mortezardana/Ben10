import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("benchmarks")


def buy_and_hold(prices):
    """
    Buy-and-hold equity curve.

    Parameters:
    - prices: pd.Series of close prices

    Returns:
    - pd.Series equity curve starting at 1.0
    """
    returns = prices.pct_change().fillna(0)
    equity = (1 + returns).cumprod()
    return equity


def random_strategy(prices, n_simulations=10000, seed=42):
    """
    Monte Carlo random entry/exit simulation.

    Parameters:
    - prices: pd.Series of close prices
    - n_simulations: number of random strategies to simulate
    - seed: random seed

    Returns:
    - dict with 'mean_return', 'std_return', 'p95_return', 'p5_return', 'equity_curves'
    """
    rng = np.random.default_rng(seed)
    returns = prices.pct_change().fillna(0).values

    final_returns = np.zeros(n_simulations)
    equity_curves = []

    for i in range(n_simulations):
        # Random signals: -1, 0, or 1
        signals = rng.choice([-1, 0, 1], size=len(returns))
        # Shift by 1 to avoid look-ahead
        positions = np.roll(signals, 1)
        positions[0] = 0

        strat_returns = positions * returns
        equity = np.cumprod(1 + strat_returns)
        final_returns[i] = equity[-1] - 1

        if i < 100:  # Store first 100 curves for visualization
            equity_curves.append(equity)

    result = {
        'mean_return': float(np.mean(final_returns)),
        'std_return': float(np.std(final_returns)),
        'p95_return': float(np.percentile(final_returns, 95)),
        'p5_return': float(np.percentile(final_returns, 5)),
        'median_return': float(np.median(final_returns)),
    }

    logger.info(f"Random strategy: mean={result['mean_return']:.4f}, "
                f"std={result['std_return']:.4f}, p5={result['p5_return']:.4f}, p95={result['p95_return']:.4f}")
    return result


def sma_crossover(prices, fast=50, slow=200):
    """
    Simple moving average crossover strategy.
    Long when fast SMA > slow SMA, flat otherwise.

    Parameters:
    - prices: pd.Series of close prices
    - fast: fast SMA period
    - slow: slow SMA period

    Returns:
    - pd.Series of signals (-1/0/+1)
    """
    fast_sma = prices.rolling(fast).mean()
    slow_sma = prices.rolling(slow).mean()

    signals = pd.Series(0, index=prices.index)
    signals[fast_sma > slow_sma] = 1
    signals[fast_sma < slow_sma] = -1

    # NaN during warmup period
    signals.iloc[:slow] = 0

    logger.info(f"SMA crossover ({fast}/{slow}): "
                f"long={int((signals==1).sum())}, short={int((signals==-1).sum())}, flat={int((signals==0).sum())}")
    return signals
