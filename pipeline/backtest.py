# pipeline/backtest.py

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger("backtest")


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame  # entry_time, exit_time, direction, entry_price, exit_price, pnl, duration
    positions: pd.Series  # position at each bar
    metrics: dict


class BacktestEngine:
    """
    Backtesting engine with transaction costs, slippage, long+short support,
    and position sizing.
    """

    def __init__(self, initial_capital=100000, commission=0.0001, slippage=0.0001):
        """
        Parameters:
        - initial_capital: starting equity
        - commission: as fraction of trade value (0.01% default for gold)
        - slippage: as fraction of price applied on position changes
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(self, prices, signals, confidence=None) -> BacktestResult:
        """
        Run backtest on price series with trading signals.

        Parameters:
        - prices: pd.Series of close prices (datetime index preferred)
        - signals: pd.Series of -1/0/+1 (short/flat/long)
        - confidence: optional pd.Series 0-1 for position sizing

        Returns:
        - BacktestResult with equity curve, trades, metrics
        """
        prices = prices.copy()
        signals = signals.copy()

        # Align
        prices, signals = prices.align(signals, join='inner')
        if confidence is not None:
            confidence = confidence.reindex(signals.index).fillna(0)

        # Execute at next bar (shift signals by 1)
        positions = signals.shift(1).fillna(0)

        # Scale by confidence if provided
        if confidence is not None:
            position_sizes = positions * confidence.shift(1).fillna(0)
        else:
            position_sizes = positions

        # Calculate bar returns
        price_returns = prices.pct_change().fillna(0)

        # Calculate costs on position changes
        position_changes = position_sizes.diff().fillna(position_sizes.iloc[0] if len(position_sizes) > 0 else 0)
        abs_changes = position_changes.abs()

        # Slippage cost: applied on position changes as fraction of price
        slippage_cost = abs_changes * self.slippage

        # Commission cost: applied on absolute position change
        commission_cost = abs_changes * self.commission

        # Strategy returns = position * price_return - costs
        strategy_returns = position_sizes * price_returns - slippage_cost - commission_cost

        # Build equity curve
        equity = self.initial_capital * (1 + strategy_returns).cumprod()

        # Extract trades
        trades = self._extract_trades(prices, positions, position_sizes, equity)

        # Compute metrics
        metrics = self._compute_metrics(equity, strategy_returns, trades)

        return BacktestResult(
            equity_curve=equity,
            returns=strategy_returns,
            trades=trades,
            positions=position_sizes,
            metrics=metrics,
        )

    def run_with_sizing(self, prices, signals, confidence, method='fixed_fractional', risk_per_trade=0.02):
        """
        Run backtest with position sizing.

        Parameters:
        - method: 'fixed_fractional', 'volatility_scaled', or 'kelly'
        - risk_per_trade: fraction of equity to risk per trade
        """
        if method == 'fixed_fractional':
            sized_confidence = confidence * risk_per_trade / risk_per_trade  # normalize
            sized_confidence = sized_confidence.clip(0, 1) * risk_per_trade / 0.02
        elif method == 'volatility_scaled':
            # Scale inversely to rolling ATR
            rolling_vol = prices.pct_change().rolling(20).std()
            target_vol = rolling_vol.median()
            vol_scale = (target_vol / rolling_vol).clip(0.1, 3.0).fillna(1.0)
            sized_confidence = confidence * vol_scale
        elif method == 'kelly':
            sized_confidence = confidence * 0.5  # half-kelly
        else:
            sized_confidence = confidence

        return self.run(prices, signals, sized_confidence)

    def _extract_trades(self, prices, positions, position_sizes, equity):
        """Extract individual trades from position changes."""
        trades_list = []
        in_trade = False
        entry_time = None
        entry_price = None
        direction = 0

        for i in range(1, len(positions)):
            curr_pos = positions.iloc[i]
            prev_pos = positions.iloc[i - 1]

            if prev_pos == 0 and curr_pos != 0:
                # Entry
                in_trade = True
                entry_time = positions.index[i]
                entry_price = prices.iloc[i]
                direction = int(np.sign(curr_pos))

            elif prev_pos != 0 and curr_pos == 0:
                # Exit
                if in_trade:
                    exit_time = positions.index[i]
                    exit_price = prices.iloc[i]
                    raw_pnl = direction * (exit_price - entry_price) / entry_price
                    # Subtract costs
                    cost = 2 * (self.commission + self.slippage)  # entry + exit
                    pnl = raw_pnl - cost

                    entry_idx = positions.index.get_loc(entry_time)
                    exit_idx = i
                    duration = exit_idx - entry_idx

                    trades_list.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'duration': duration,
                    })
                    in_trade = False

            elif prev_pos != 0 and curr_pos != 0 and np.sign(prev_pos) != np.sign(curr_pos):
                # Direction flip — close + open
                if in_trade:
                    exit_time = positions.index[i]
                    exit_price = prices.iloc[i]
                    raw_pnl = direction * (exit_price - entry_price) / entry_price
                    cost = 2 * (self.commission + self.slippage)
                    pnl = raw_pnl - cost

                    entry_idx = positions.index.get_loc(entry_time)
                    duration = i - entry_idx

                    trades_list.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'duration': duration,
                    })

                # New entry
                entry_time = positions.index[i]
                entry_price = prices.iloc[i]
                direction = int(np.sign(curr_pos))
                in_trade = True

        # Close open trade at end
        if in_trade and len(positions) > 0:
            exit_time = positions.index[-1]
            exit_price = prices.iloc[-1]
            raw_pnl = direction * (exit_price - entry_price) / entry_price
            cost = 2 * (self.commission + self.slippage)
            pnl = raw_pnl - cost
            entry_idx = positions.index.get_loc(entry_time)
            duration = len(positions) - 1 - entry_idx

            trades_list.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'duration': duration,
            })

        if trades_list:
            return pd.DataFrame(trades_list)
        return pd.DataFrame(columns=['entry_time', 'exit_time', 'direction',
                                     'entry_price', 'exit_price', 'pnl', 'duration'])

    def _compute_metrics(self, equity, returns, trades):
        """Compute basic metrics inline (doesn't depend on utils/metrics.py)."""
        metrics = {}
        try:
            from utils.metrics import compute_all_metrics
            metrics = compute_all_metrics(equity, returns, trades)
        except ImportError:
            # Fallback basic metrics
            total_return = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 0 else 0
            metrics['total_return'] = float(total_return)
            metrics['trade_count'] = len(trades)
            if len(returns) > 1:
                metrics['sharpe_ratio'] = float(returns.mean() / returns.std() * np.sqrt(1512)) if returns.std() > 0 else 0
            if len(equity) > 1:
                running_max = equity.cummax()
                dd = (equity - running_max) / running_max
                metrics['max_drawdown'] = float(abs(dd.min()))
            if len(trades) > 0 and 'pnl' in trades.columns:
                metrics['win_rate'] = float((trades['pnl'] > 0).sum() / len(trades))

        return metrics


# ---------------------------------------------------------------------------
# Legacy function kept for backward compatibility
# ---------------------------------------------------------------------------

def simple_strategy_backtest(df, entry_col='prediction', price_col='Close'):
    """Original simple backtest. Kept for backward compatibility."""
    logger.info("Running simple strategy backtest")
    df = df.copy()

    df['position'] = df[entry_col].shift(1).fillna(0)
    df['log_return'] = (df[price_col] / df[price_col].shift(1)).apply(lambda x: np.nan if x <= 0 else np.log(x))
    df['strategy_return'] = df['position'] * df['log_return']
    df['equity_curve'] = df['strategy_return'].cumsum().apply(np.exp)

    keep_cols = ['target', 'prediction', 'position', 'log_return', 'strategy_return', 'equity_curve']
    keep_cols = [col for col in keep_cols if col in df.columns]

    logger.info(f"Total return: {df['equity_curve'].iloc[-1] - 1:.2%}")
    return df[keep_cols].dropna()
