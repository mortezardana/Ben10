"""
Comprehensive trading performance metrics module.

All annualization assumes 4H bar data:
    BARS_PER_YEAR = 6 bars/day * 252 trading days = 1512 bars/year

Functions accept pandas Series / DataFrames and return scalars or dicts.
Edge cases (empty inputs, zero division) return 0.0 or float('nan') instead
of raising exceptions.
"""

import math
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BARS_PER_DAY = 6          # 4-hour bars per trading day
TRADING_DAYS = 252         # trading days per year
BARS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS  # 1512


# ===========================================================================
# Core Metrics
# ===========================================================================

def profit_factor(trades: pd.DataFrame) -> float:
    """Ratio of gross profit to gross loss.

    Parameters
    ----------
    trades : pd.DataFrame
        Must contain a ``pnl`` column with per-trade profit/loss values.

    Returns
    -------
    float
        gross_profit / gross_loss.  Returns 0.0 when *trades* is empty or
        there are no losses (infinite profit factor is capped to 0.0 to
        avoid downstream issues — callers may choose to treat 0.0 as inf).
    """
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return 0.0
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough percentage decline in an equity curve.

    Parameters
    ----------
    equity : pd.Series
        Cumulative equity values (e.g. starting at 1.0).

    Returns
    -------
    float
        Maximum drawdown expressed as a positive fraction (e.g. 0.25 = 25%).
        Returns 0.0 for empty or constant equity.
    """
    if equity is None or len(equity) < 2:
        return 0.0
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    drawdowns = drawdowns.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    mdd = drawdowns.min()
    return float(abs(mdd))


def max_drawdown_duration(equity: pd.Series) -> int:
    """Number of bars in the longest drawdown period.

    A drawdown period starts when equity drops below its running maximum
    and ends when equity recovers to a new high.

    Parameters
    ----------
    equity : pd.Series
        Cumulative equity values.

    Returns
    -------
    int
        Length (in bars) of the longest drawdown period.  Returns 0 for
        empty or monotonically increasing equity.
    """
    if equity is None or len(equity) < 2:
        return 0
    running_max = equity.cummax()
    in_drawdown = equity < running_max

    longest = 0
    current = 0
    for flag in in_drawdown:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def calmar_ratio(equity: pd.Series) -> float:
    """Calmar ratio: annualized return divided by maximum drawdown.

    Parameters
    ----------
    equity : pd.Series
        Cumulative equity values.

    Returns
    -------
    float
        calmar = annualized_return / max_drawdown.
        Returns 0.0 when drawdown is zero or equity is too short.
    """
    ann_ret = annualized_return(equity)
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(ann_ret / mdd)


def sortino_ratio(returns: pd.Series, target: float = 0) -> float:
    """Sortino ratio: excess return over target divided by downside deviation.

    Parameters
    ----------
    returns : pd.Series
        Per-bar returns.
    target : float, default 0
        Minimum acceptable return per bar.

    Returns
    -------
    float
        Annualized Sortino ratio.  Returns 0.0 when downside deviation is
        zero or returns are empty.

    Formula
    -------
    downside = sqrt(mean(min(r - target, 0)^2))
    sortino  = (mean(r) - target) / downside * sqrt(BARS_PER_YEAR)
    """
    if returns is None or len(returns) < 2:
        return 0.0
    excess = returns - target
    downside = excess.copy()
    downside[downside > 0] = 0.0
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return 0.0
    mean_excess = excess.mean()
    return float((mean_excess / downside_dev) * np.sqrt(BARS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, rf: float = 0) -> float:
    """Annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Per-bar returns.
    rf : float, default 0
        Risk-free rate **per bar**.

    Returns
    -------
    float
        sharpe = (mean(r) - rf) / std(r) * sqrt(BARS_PER_YEAR).
        Returns 0.0 when standard deviation is zero or returns are empty.
    """
    if returns is None or len(returns) < 2:
        return 0.0
    excess = returns - rf
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(BARS_PER_YEAR))


def win_rate(trades: pd.DataFrame) -> float:
    """Percentage of profitable trades.

    Parameters
    ----------
    trades : pd.DataFrame
        Must contain a ``pnl`` column.

    Returns
    -------
    float
        Fraction of trades where pnl > 0 (e.g. 0.55 = 55 %).
        Returns 0.0 for empty input.
    """
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return 0.0
    n = len(trades)
    if n == 0:
        return 0.0
    return float((trades["pnl"] > 0).sum() / n)


def avg_win_loss_ratio(trades: pd.DataFrame) -> float:
    """Average win divided by average loss (absolute value).

    Parameters
    ----------
    trades : pd.DataFrame
        Must contain a ``pnl`` column.

    Returns
    -------
    float
        mean(winning pnl) / |mean(losing pnl)|.
        Returns 0.0 when there are no wins or no losses.
    """
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return 0.0
    wins = trades.loc[trades["pnl"] > 0, "pnl"]
    losses = trades.loc[trades["pnl"] < 0, "pnl"]
    if wins.empty or losses.empty:
        return 0.0
    avg_loss = abs(losses.mean())
    if avg_loss == 0:
        return 0.0
    return float(wins.mean() / avg_loss)


def expectancy(trades: pd.DataFrame) -> float:
    """Expected value per trade.

    Parameters
    ----------
    trades : pd.DataFrame
        Must contain a ``pnl`` column.

    Returns
    -------
    float
        (win_rate * avg_win) - (loss_rate * avg_loss).
        Returns 0.0 for empty input.

    Notes
    -----
    avg_loss is taken as a positive number so the subtraction gives the
    net expected value.
    """
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return 0.0
    wins = trades.loc[trades["pnl"] > 0, "pnl"]
    losses = trades.loc[trades["pnl"] < 0, "pnl"]
    n = len(trades)
    if n == 0:
        return 0.0
    wr = len(wins) / n
    lr = len(losses) / n
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0
    return float(wr * avg_win - lr * avg_loss)


def trade_count(trades: pd.DataFrame) -> int:
    """Total number of trades.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade log (any schema accepted; only row count matters).

    Returns
    -------
    int
    """
    if trades is None or trades.empty:
        return 0
    return int(len(trades))


def max_consecutive_losses(trades: pd.DataFrame) -> int:
    """Longest consecutive losing streak.

    Parameters
    ----------
    trades : pd.DataFrame
        Must contain a ``pnl`` column.

    Returns
    -------
    int
        Number of consecutive trades with pnl < 0.  Returns 0 for empty
        input or when there are no losing trades.
    """
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return 0
    is_loss = (trades["pnl"] < 0).astype(int)
    longest = 0
    current = 0
    for val in is_loss:
        if val:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def annualized_return(equity: pd.Series) -> float:
    """Compound annual growth rate (CAGR) derived from an equity curve.

    Parameters
    ----------
    equity : pd.Series
        Cumulative equity (e.g. starting at 1.0).

    Returns
    -------
    float
        CAGR as a decimal (e.g. 0.12 = 12 % per year).
        Returns 0.0 when the equity series is too short or starts at zero.

    Formula
    -------
    total_return = equity[-1] / equity[0]
    n_years      = len(equity) / BARS_PER_YEAR
    CAGR         = total_return^(1 / n_years) - 1
    """
    if equity is None or len(equity) < 2:
        return 0.0
    start = equity.iloc[0]
    end = equity.iloc[-1]
    if start == 0:
        return 0.0
    total_return = end / start
    if total_return <= 0:
        return 0.0
    n_bars = len(equity)
    n_years = n_bars / BARS_PER_YEAR
    if n_years == 0:
        return 0.0
    return float(total_return ** (1.0 / n_years) - 1.0)


def volatility(returns: pd.Series) -> float:
    """Annualized volatility (standard deviation of returns).

    Parameters
    ----------
    returns : pd.Series
        Per-bar returns.

    Returns
    -------
    float
        Annualized standard deviation = std(returns) * sqrt(BARS_PER_YEAR).
        Returns 0.0 for empty input.
    """
    if returns is None or len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(BARS_PER_YEAR))


# ===========================================================================
# Statistical Metrics
# ===========================================================================

def statistical_significance(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """Binomial test for prediction accuracy.

    Tests whether the observed accuracy is significantly better than random
    (50 %) guessing.

    Parameters
    ----------
    predictions : array-like
        Binary predictions (0 or 1).
    actuals : array-like
        Binary ground-truth labels (0 or 1).

    Returns
    -------
    dict
        ``{"accuracy": float, "p_value": float, "n_obs": int,
           "significant_at_05": bool}``
        Returns ``{"accuracy": 0.0, "p_value": 1.0, "n_obs": 0,
        "significant_at_05": False}`` for empty inputs.
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    n = len(predictions)
    if n == 0 or len(actuals) == 0 or n != len(actuals):
        return {"accuracy": 0.0, "p_value": 1.0, "n_obs": 0,
                "significant_at_05": False}
    correct = int((predictions == actuals).sum())
    accuracy = correct / n
    # Two-sided binomial test: H0 accuracy = 0.5
    result = stats.binomtest(correct, n, p=0.5, alternative="greater")
    p_value = float(result.pvalue)
    return {
        "accuracy": float(accuracy),
        "p_value": p_value,
        "n_obs": n,
        "significant_at_05": p_value < 0.05,
    }


def bootstrap_confidence_interval(
    returns: pd.Series, n: int = 10000, ci: float = 0.95
) -> tuple:
    """Bootstrap confidence interval for the mean return.

    Parameters
    ----------
    returns : pd.Series or array-like
        Per-bar returns.
    n : int, default 10_000
        Number of bootstrap resamples.
    ci : float, default 0.95
        Confidence level (e.g. 0.95 for 95 %).

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) of the mean return.
        Returns (0.0, 0.0) for empty input.
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)  # reproducible
    boot_means = np.empty(n)
    size = len(returns)
    for i in range(n):
        sample = rng.choice(returns, size=size, replace=True)
        boot_means[i] = sample.mean()
    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lower, upper)


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Bailey & Lopez de Prado's Deflated Sharpe Ratio (DSR).

    Adjusts an observed Sharpe ratio for the number of trials (strategies
    tested), non-normal returns, and finite sample size.

    Parameters
    ----------
    sharpe : float
        Observed (annualized) Sharpe ratio of the selected strategy.
    n_trials : int
        Total number of strategies / parameter combos tried.
    n_obs : int
        Number of return observations used to compute *sharpe*.
    skew : float
        Skewness of the return series.
    kurtosis : float
        *Excess* kurtosis of the return series.

    Returns
    -------
    float
        Probability (0-1) that the true Sharpe ratio is positive after
        adjusting for selection bias.  Values near 1.0 indicate a robust
        Sharpe; values below 0.5 are suspect.  Returns 0.0 when inputs
        are degenerate.

    Reference
    ---------
    Bailey, D.H. and Lopez de Prado, M. (2014). "The Deflated Sharpe
    Ratio: Correcting for Selection Bias, Back-test Over-fitting, and
    Non-Normality."  Journal of Portfolio Management.
    """
    if n_trials < 1 or n_obs < 2:
        return 0.0

    # Expected maximum Sharpe ratio under the null (all strategies have
    # SR = 0) given n_trials independent trials.
    # E[max(Z)] ≈ (1 - gamma) * Phi_inv(1 - 1/N) + gamma * Phi_inv(1 - 1/(N*e))
    # Simplified approximation using Euler-Mascheroni:
    euler_mascheroni = 0.5772156649
    try:
        sr_max = (
            (1 - euler_mascheroni) * stats.norm.ppf(1 - 1.0 / n_trials)
            + euler_mascheroni * stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
        )
    except Exception:
        return 0.0

    # Adjusted standard deviation of the Sharpe estimator accounting for
    # skew and kurtosis (Lo 2002 / Bailey & LdP 2014):
    # Var(SR) ≈ (1 - skew*SR + (kurtosis-1)/4 * SR^2) / (n_obs - 1)
    sr = sharpe
    var_sr = (1.0 - skew * sr + ((kurtosis - 1) / 4.0) * sr ** 2) / (n_obs - 1)
    if var_sr <= 0:
        return 0.0
    std_sr = math.sqrt(var_sr)

    # DSR = Phi( (SR - SR_max) / std(SR) )
    if std_sr == 0:
        return 0.0
    z = (sr - sr_max) / std_sr
    dsr = float(stats.norm.cdf(z))
    return dsr


# ===========================================================================
# Aggregate
# ===========================================================================

def compute_all_metrics(
    equity: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame,
) -> dict:
    """Compute every metric in this module and return them in a single dict.

    Parameters
    ----------
    equity : pd.Series
        Cumulative equity curve.
    returns : pd.Series
        Per-bar returns.
    trades : pd.DataFrame
        Trade log with at least a ``pnl`` column.

    Returns
    -------
    dict
        Keys mirror the function names; values are the computed metrics.
    """
    return {
        # Core
        "profit_factor": profit_factor(trades),
        "max_drawdown": max_drawdown(equity),
        "max_drawdown_duration": max_drawdown_duration(equity),
        "calmar_ratio": calmar_ratio(equity),
        "sortino_ratio": sortino_ratio(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "win_rate": win_rate(trades),
        "avg_win_loss_ratio": avg_win_loss_ratio(trades),
        "expectancy": expectancy(trades),
        "trade_count": trade_count(trades),
        "max_consecutive_losses": max_consecutive_losses(trades),
        "annualized_return": annualized_return(equity),
        "volatility": volatility(returns),
        # Statistical
        "bootstrap_ci_95": bootstrap_confidence_interval(returns),
    }
