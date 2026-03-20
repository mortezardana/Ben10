"""Validation framework for trading strategies."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger("validation")


@dataclass
class ValidationResult:
    per_split_metrics: list  # metrics for each walk-forward window
    aggregate: dict           # mean, std, min, max of each metric
    walk_forward_efficiency: float  # OOS return / IS return
    equity_curves: list      # per-split equity curves


class WalkForwardValidator:
    """Walk-forward validation with rolling and expanding windows."""

    def __init__(self, n_splits=5, train_size=None, test_size=None, expanding=False, gap=0):
        """
        Parameters:
        - n_splits: number of walk-forward windows
        - train_size: fixed training window size (bars). None = auto-calculated
        - test_size: test window size (bars). None = auto-calculated
        - expanding: if True, training window grows; if False, rolls forward
        - gap: embargo gap between train and test (bars) to prevent leakage
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.expanding = expanding
        self.gap = gap

    def split(self, X):
        """Generate train/test index arrays for each split."""
        n = len(X)

        if self.test_size is None:
            test_size = n // (self.n_splits + 1)
        else:
            test_size = self.test_size

        if self.train_size is None:
            train_size = n - (self.n_splits * test_size) - (self.n_splits * self.gap)
            train_size = max(train_size, test_size * 2)  # Minimum 2x test
        else:
            train_size = self.train_size

        splits = []
        for i in range(self.n_splits):
            if self.expanding:
                train_start = 0
                train_end = train_size + i * test_size
            else:
                train_start = i * test_size
                train_end = train_start + train_size

            test_start = train_end + self.gap
            test_end = test_start + test_size

            if test_end > n:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))

        return splits

    def validate(self, pipeline, data, prices=None, target_col='target'):
        """
        Run full walk-forward validation.

        Parameters:
        - pipeline: TradingPipeline instance
        - data: full DataFrame
        - prices: price Series for backtesting (optional)
        - target_col: name of target column

        Returns:
        - ValidationResult
        """
        splits = self.split(data)
        per_split_metrics = []
        equity_curves = []
        is_returns = []
        oos_returns = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Walk-forward fold {fold_idx + 1}/{len(splits)}: "
                       f"train={len(train_idx)}, test={len(test_idx)}")

            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # Train
            train_metrics = pipeline.train(train_data)

            # Evaluate on train (IS)
            is_metrics = pipeline.evaluate(train_data)

            # Evaluate on test (OOS)
            oos_metrics = pipeline.evaluate(test_data)

            is_returns.append(is_metrics.get('accuracy', 0))
            oos_returns.append(oos_metrics.get('accuracy', 0))

            per_split_metrics.append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'is_metrics': is_metrics,
                'oos_metrics': oos_metrics,
            })

        # Aggregate
        oos_accuracies = [m['oos_metrics'].get('accuracy', 0) for m in per_split_metrics]
        is_accuracies = [m['is_metrics'].get('accuracy', 0) for m in per_split_metrics]

        aggregate = {
            'mean_oos_accuracy': float(np.mean(oos_accuracies)) if oos_accuracies else 0,
            'std_oos_accuracy': float(np.std(oos_accuracies)) if oos_accuracies else 0,
            'min_oos_accuracy': float(np.min(oos_accuracies)) if oos_accuracies else 0,
            'max_oos_accuracy': float(np.max(oos_accuracies)) if oos_accuracies else 0,
            'mean_is_accuracy': float(np.mean(is_accuracies)) if is_accuracies else 0,
        }

        # Walk-forward efficiency
        mean_is = np.mean(is_returns) if is_returns else 0
        mean_oos = np.mean(oos_returns) if oos_returns else 0
        wfe = mean_oos / mean_is if mean_is > 0 else 0

        logger.info(f"Walk-forward efficiency: {wfe:.3f}")

        return ValidationResult(
            per_split_metrics=per_split_metrics,
            aggregate=aggregate,
            walk_forward_efficiency=float(wfe),
            equity_curves=equity_curves,
        )
