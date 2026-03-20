"""Risk management: confidence filtering, position sizing, drawdown circuit breaker."""
import numpy as np
import pandas as pd
from shared.interfaces import PipelineOutput
from utils.logger import get_logger

logger = get_logger("risk")


class RiskManager:
    def __init__(self, confidence_threshold=0.6, max_position=1.0, risk_per_trade=0.02):
        self.confidence_threshold = confidence_threshold
        self.max_position = max_position
        self.risk_per_trade = risk_per_trade

    def apply(self, combined_output, atr=None, equity=None):
        """
        Apply risk filters to combined output.

        Returns DataFrame with: signal, position_size, stop_loss, take_profit
        """
        signals = combined_output.signals.copy()
        confidence = combined_output.confidence.copy()

        # Filter by confidence threshold
        signals[confidence < self.confidence_threshold] = 0

        # Position sizing based on confidence
        position_size = confidence * self.max_position
        position_size[signals == 0] = 0

        result = pd.DataFrame(index=signals.index)
        result['signal'] = signals
        result['position_size'] = position_size

        # ATR-based stops
        if atr is not None:
            atr_aligned = atr.reindex(signals.index).fillna(atr.median())
            result['stop_loss'] = 1.0 * atr_aligned  # 1x ATR stop
            result['take_profit'] = 2.0 * atr_aligned  # 2x ATR target

        filtered = int((combined_output.signals != 0).sum() - (signals != 0).sum())
        logger.info(f"Risk filter: {filtered} signals filtered by confidence < {self.confidence_threshold}")
        return result

    def volatility_target(self, atr, target_vol=0.10):
        """Scale position inversely to ATR for volatility targeting."""
        median_atr = atr.median()
        if median_atr == 0:
            return pd.Series(1.0, index=atr.index)
        scale = (median_atr / atr).clip(0.1, 3.0)
        return scale * target_vol

    def half_kelly(self, win_rate, avg_win_loss_ratio):
        """Kelly criterion at 50% for conservative sizing."""
        if avg_win_loss_ratio == 0:
            return 0.0
        kelly = win_rate - (1 - win_rate) / avg_win_loss_ratio
        return max(0, kelly * 0.5)  # Half Kelly

    def drawdown_check(self, equity_curve, max_drawdown=0.15):
        """Returns True if current drawdown exceeds threshold (circuit breaker)."""
        if equity_curve is None or len(equity_curve) < 2:
            return False
        peak = equity_curve.cummax()
        current_dd = (equity_curve.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1]
        triggered = abs(current_dd) > max_drawdown
        if triggered:
            logger.warning(f"CIRCUIT BREAKER: Drawdown {abs(current_dd):.1%} > {max_drawdown:.1%}")
        return triggered
