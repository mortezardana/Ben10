"""Validation framework for trading strategies."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger("validation")


@dataclass
class ValidationResult:
    per_split_metrics: list    # metrics for each walk-forward window
    aggregate: dict            # mean, std, min, max of each metric
    walk_forward_efficiency: float  # OOS accuracy / IS accuracy
    equity_curves: list        # per-split OOS equity curves
    combined_oos_equity: pd.Series = None  # stitched OOS equity across all folds


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
            train_size = max(train_size, test_size * 2)
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

    def validate(self, pipeline, data, prices=None, target_col='target',
                 backtest_engine=None):
        """
        Run full walk-forward validation.

        For each fold: train on train window, predict+evaluate on test window,
        optionally backtest on test window.

        Parameters:
        - pipeline: TradingPipeline instance
        - data: full DataFrame (unnormalized — each fold normalizes independently)
        - prices: price Series for backtesting (optional, same index as data)
        - target_col: name of target column
        - backtest_engine: BacktestEngine instance (optional)

        Returns:
        - ValidationResult
        """
        splits = self.split(data)
        per_split_metrics = []
        equity_curves = []
        all_oos_returns = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Walk-forward fold {fold_idx + 1}/{len(splits)}: "
                        f"train[{train_idx[0]}:{train_idx[-1]}] "
                        f"test[{test_idx[0]}:{test_idx[-1]}]")

            train_data = data.iloc[train_idx].copy()
            test_data = data.iloc[test_idx].copy()

            # Normalize per fold — fit on train only
            from sklearn.preprocessing import StandardScaler
            exclude = [target_col, 'date', 'Date']
            feat_cols = [c for c in train_data.columns
                         if c not in exclude
                         and train_data[c].dtype in ['float64', 'float32', 'int64', 'int32']]

            scaler = StandardScaler()
            scaler.fit(train_data[feat_cols])
            train_data[feat_cols] = scaler.transform(train_data[feat_cols])
            test_data[feat_cols] = scaler.transform(test_data[feat_cols])

            # Train
            train_metrics = pipeline.train(train_data)

            # Evaluate IS
            is_metrics = pipeline.evaluate(train_data)

            # Evaluate OOS
            oos_metrics = pipeline.evaluate(test_data)

            # Backtest OOS if engine and prices provided
            bt_metrics = {}
            oos_equity = None
            if backtest_engine is not None and prices is not None:
                try:
                    output = pipeline.predict(test_data)
                    test_prices = prices.iloc[test_idx]
                    # Align signals with prices
                    signals = output.signals.reindex(test_prices.index).fillna(0)
                    bt_result = backtest_engine.run(test_prices, signals)
                    bt_metrics = bt_result.metrics
                    oos_equity = bt_result.equity_curve
                    equity_curves.append(oos_equity)
                    all_oos_returns.append(bt_result.returns)
                except Exception as e:
                    logger.warning(f"Fold {fold_idx + 1} backtest failed: {e}")

            fold_result = {
                'fold': fold_idx,
                'train_range': (int(train_idx[0]), int(train_idx[-1])),
                'test_range': (int(test_idx[0]), int(test_idx[-1])),
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'is_metrics': is_metrics,
                'oos_metrics': oos_metrics,
                'bt_metrics': bt_metrics,
            }
            per_split_metrics.append(fold_result)

            # Log fold summary
            is_acc = is_metrics.get('accuracy', 0)
            oos_acc = oos_metrics.get('accuracy', 0)
            sharpe = bt_metrics.get('sharpe_ratio', 'N/A')
            logger.info(f"  IS acc={is_acc:.4f}  OOS acc={oos_acc:.4f}  Sharpe={sharpe}")

        # Aggregate
        oos_accuracies = [m['oos_metrics'].get('accuracy', 0) for m in per_split_metrics]
        is_accuracies = [m['is_metrics'].get('accuracy', 0) for m in per_split_metrics]
        oos_sharpes = [m['bt_metrics'].get('sharpe_ratio', 0) for m in per_split_metrics if m['bt_metrics']]

        aggregate = {
            'mean_oos_accuracy': float(np.mean(oos_accuracies)),
            'std_oos_accuracy': float(np.std(oos_accuracies)),
            'min_oos_accuracy': float(np.min(oos_accuracies)),
            'max_oos_accuracy': float(np.max(oos_accuracies)),
            'mean_is_accuracy': float(np.mean(is_accuracies)),
            'std_is_accuracy': float(np.std(is_accuracies)),
        }

        if oos_sharpes:
            aggregate['mean_oos_sharpe'] = float(np.mean(oos_sharpes))
            aggregate['std_oos_sharpe'] = float(np.std(oos_sharpes))
            aggregate['min_oos_sharpe'] = float(np.min(oos_sharpes))
            aggregate['max_oos_sharpe'] = float(np.max(oos_sharpes))

        # Walk-forward efficiency = OOS accuracy / IS accuracy
        mean_is = np.mean(is_accuracies) if is_accuracies else 0
        mean_oos = np.mean(oos_accuracies) if oos_accuracies else 0
        wfe = mean_oos / mean_is if mean_is > 0 else 0

        # Stitch OOS equity curves
        combined_equity = None
        if all_oos_returns:
            combined = pd.concat(all_oos_returns)
            combined_equity = (1 + combined).cumprod() * 100000  # start at 100k

        logger.info(f"\n{'='*50}")
        logger.info(f"WALK-FORWARD SUMMARY ({len(splits)} folds)")
        logger.info(f"{'='*50}")
        logger.info(f"IS accuracy:  {mean_is:.4f} +/- {np.std(is_accuracies):.4f}")
        logger.info(f"OOS accuracy: {mean_oos:.4f} +/- {np.std(oos_accuracies):.4f}")
        logger.info(f"WF efficiency: {wfe:.3f} (>0.7 = good transferability)")
        if oos_sharpes:
            logger.info(f"OOS Sharpe:   {np.mean(oos_sharpes):.3f} +/- {np.std(oos_sharpes):.3f}")

        return ValidationResult(
            per_split_metrics=per_split_metrics,
            aggregate=aggregate,
            walk_forward_efficiency=float(wfe),
            equity_curves=equity_curves,
            combined_oos_equity=combined_equity,
        )
