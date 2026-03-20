"""Advanced validation: CPCV and Monte Carlo robustness testing."""

import numpy as np
import pandas as pd
from itertools import combinations
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger("validation_advanced")


@dataclass
class CPCVResult:
    per_path_metrics: list
    aggregate: dict
    pbo: float  # Probability of Backtest Overfitting


class CPCValidator:
    """Combinatorial Purged Cross-Validation (Lopez de Prado)."""

    def __init__(self, n_groups=6, n_test_groups=2, purge_gap=6, embargo_gap=12):
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap

    def split(self, X):
        """Generate all valid train/test combinations."""
        n = len(X)
        group_size = n // self.n_groups

        # Create group boundaries
        groups = []
        for i in range(self.n_groups):
            start = i * group_size
            end = start + group_size if i < self.n_groups - 1 else n
            groups.append(np.arange(start, end))

        # Generate all C(n_groups, n_test_groups) test combinations
        splits = []
        for test_combo in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.concatenate([groups[g] for g in test_combo])

            # Train = all groups NOT in test, with purging
            train_groups = [g for g in range(self.n_groups) if g not in test_combo]
            train_idx = np.concatenate([groups[g] for g in train_groups])

            # Purge: remove bars near train/test boundaries
            test_set = set(test_idx)
            purge_mask = np.ones(len(train_idx), dtype=bool)
            for ti in range(len(train_idx)):
                idx = train_idx[ti]
                # Check if within purge_gap + embargo_gap of any test index
                for gap in range(1, self.purge_gap + self.embargo_gap + 1):
                    if (idx + gap) in test_set or (idx - gap) in test_set:
                        purge_mask[ti] = False
                        break

            train_idx_purged = train_idx[purge_mask]
            if len(train_idx_purged) > 0 and len(test_idx) > 0:
                splits.append((train_idx_purged, test_idx))

        return splits

    def validate(self, pipeline, data, target_col='target'):
        """Run all CPCV paths and aggregate."""
        splits = self.split(data)
        per_path_metrics = []
        is_better_count = 0

        for path_idx, (train_idx, test_idx) in enumerate(splits):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            pipeline.train(train_data)
            is_metrics = pipeline.evaluate(train_data)
            oos_metrics = pipeline.evaluate(test_data)

            is_acc = is_metrics.get('accuracy', 0)
            oos_acc = oos_metrics.get('accuracy', 0)

            if is_acc > oos_acc:
                is_better_count += 1

            per_path_metrics.append({
                'path': path_idx,
                'is_accuracy': is_acc,
                'oos_accuracy': oos_acc,
            })

        n_paths = len(per_path_metrics)
        pbo = is_better_count / n_paths if n_paths > 0 else 1.0

        oos_accs = [m['oos_accuracy'] for m in per_path_metrics]
        aggregate = {
            'n_paths': n_paths,
            'mean_oos_accuracy': float(np.mean(oos_accs)) if oos_accs else 0,
            'std_oos_accuracy': float(np.std(oos_accs)) if oos_accs else 0,
            'pbo': pbo,
        }

        logger.info(f"CPCV: {n_paths} paths, PBO={pbo:.3f}, mean OOS acc={aggregate['mean_oos_accuracy']:.4f}")

        return CPCVResult(per_path_metrics=per_path_metrics, aggregate=aggregate, pbo=pbo)

    def probability_of_backtest_overfitting(self):
        """Access PBO from last validation run."""
        return getattr(self, '_last_pbo', None)


class MonteCarloValidator:
    """Monte Carlo simulation for strategy robustness testing."""

    def __init__(self, n_simulations=10000, seed=42):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    def test_trade_shuffle(self, trades):
        """Shuffle trade order, recompute equity curves."""
        if trades is None or trades.empty or 'pnl' not in trades.columns:
            return {'error': 'No valid trades'}

        pnls = trades['pnl'].values
        final_returns = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            shuffled = self.rng.permutation(pnls)
            equity = np.cumprod(1 + shuffled)
            final_returns[i] = equity[-1] - 1

        return {
            'mean_return': float(np.mean(final_returns)),
            'std_return': float(np.std(final_returns)),
            'p5': float(np.percentile(final_returns, 5)),
            'p95': float(np.percentile(final_returns, 95)),
            'original_return': float(np.cumprod(1 + pnls)[-1] - 1),
        }

    def test_skip_trades(self, trades, skip_rate=0.1):
        """Randomly skip X% of trades, recompute metrics."""
        if trades is None or trades.empty or 'pnl' not in trades.columns:
            return {'error': 'No valid trades'}

        pnls = trades['pnl'].values
        n_trades = len(pnls)
        n_skip = max(1, int(n_trades * skip_rate))

        final_returns = np.zeros(self.n_simulations)
        for i in range(self.n_simulations):
            mask = np.ones(n_trades, dtype=bool)
            skip_idx = self.rng.choice(n_trades, size=n_skip, replace=False)
            mask[skip_idx] = False
            kept = pnls[mask]
            if len(kept) > 0:
                equity = np.cumprod(1 + kept)
                final_returns[i] = equity[-1] - 1

        return {
            'mean_return': float(np.mean(final_returns)),
            'std_return': float(np.std(final_returns)),
            'p5': float(np.percentile(final_returns, 5)),
            'p95': float(np.percentile(final_returns, 95)),
            'skip_rate': skip_rate,
        }

    def test_bootstrap_returns(self, returns):
        """Bootstrap resample returns for confidence intervals."""
        returns = np.asarray(returns)
        if len(returns) == 0:
            return {'error': 'No returns'}

        boot_means = np.zeros(self.n_simulations)
        boot_sharpes = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            sample = self.rng.choice(returns, size=len(returns), replace=True)
            boot_means[i] = sample.mean()
            std = sample.std()
            boot_sharpes[i] = sample.mean() / std * np.sqrt(1512) if std > 0 else 0

        return {
            'mean_return_ci': (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))),
            'sharpe_ci': (float(np.percentile(boot_sharpes, 2.5)), float(np.percentile(boot_sharpes, 97.5))),
        }

    def full_robustness_report(self, trades, returns):
        """Run all tests and return comprehensive report."""
        report = {
            'trade_shuffle': self.test_trade_shuffle(trades),
            'skip_trades': self.test_skip_trades(trades),
            'bootstrap_returns': self.test_bootstrap_returns(returns),
        }
        logger.info("Monte Carlo robustness report generated")
        return report
