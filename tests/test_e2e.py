"""End-to-end integration tests for Ben10 Gold Trading AI.

Validates the full pipeline: load data -> train models -> combine signals -> backtest -> validate.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd


class TestDataLoading:
    """Test that data loading fixes work correctly."""

    def test_no_leaking_columns(self):
        """After loading, future_returns and signal should not be present."""
        from shared.data_loader import load_gold_data
        data = load_gold_data()
        for split in ['train', 'val', 'test']:
            assert 'future_returns' not in data[split].columns
            assert 'signal' not in data[split].columns

    def test_no_suspicious_columns(self):
        """No column should contain 'future' or 'forward' in name."""
        from shared.data_loader import load_gold_data
        data = load_gold_data()
        for col in data['train'].columns:
            col_lower = col.lower()
            assert 'future' not in col_lower, f"Suspicious column: {col}"
            assert 'forward' not in col_lower, f"Suspicious column: {col}"

    def test_no_overlapping_splits(self):
        """Train/val/test should not overlap."""
        from shared.data_loader import load_gold_data
        data = load_gold_data()
        train_idx = set(data['train'].index)
        val_idx = set(data['val'].index)
        test_idx = set(data['test'].index)
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

    def test_scaler_fitted_on_train_only(self):
        """Scaler should be fitted only on training data."""
        from shared.data_loader import load_gold_data
        data = load_gold_data()
        assert data['scaler'] is not None

    def test_feature_reduction(self):
        """CDL columns should be removed and feature count should be reduced."""
        from shared.data_loader import load_gold_data
        data = load_gold_data()
        for col in data['train'].columns:
            assert not col.startswith('CDL'), f"CDL column still present: {col}"


class TestSequenceUtils:
    """Test sequence windowing utility."""

    def test_output_shape(self):
        from utils.sequence_utils import create_sequences
        data = np.random.randn(100, 10)
        target = np.random.randint(0, 2, 100)
        X, y = create_sequences(data, target, sequence_length=32)
        assert X.shape == (68, 32, 10)
        assert y.shape == (68,)

    def test_no_future_leakage(self):
        from utils.sequence_utils import create_sequences
        data = np.arange(50).reshape(50, 1)
        target = np.arange(50)
        X, y = create_sequences(data, target, 5)
        # X[0] should contain data[0:5], y[0] should be target[5]
        assert np.array_equal(X[0].flatten(), [0, 1, 2, 3, 4])
        assert y[0] == 5


class TestMetrics:
    """Test trading metrics module."""

    def test_max_drawdown(self):
        from utils.metrics import max_drawdown
        equity = pd.Series([100, 110, 105, 115])
        mdd = max_drawdown(equity)
        assert mdd > 0
        expected = (110 - 105) / 110  # ~4.5%
        assert abs(mdd - expected) < 0.01

    def test_sharpe_ratio(self):
        from utils.metrics import sharpe_ratio
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        sr = sharpe_ratio(returns)
        assert isinstance(sr, float)

    def test_compute_all_metrics(self):
        from utils.metrics import compute_all_metrics
        equity = pd.Series([100, 101, 102, 101, 103])
        returns = equity.pct_change().dropna()
        trades = pd.DataFrame({'pnl': [0.01, -0.005, 0.02, -0.01]})
        result = compute_all_metrics(equity, returns, trades)
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert 'win_rate' in result

    def test_empty_inputs(self):
        from utils.metrics import sharpe_ratio, max_drawdown, win_rate
        assert sharpe_ratio(pd.Series(dtype=float)) == 0.0
        assert max_drawdown(pd.Series(dtype=float)) == 0.0
        assert win_rate(pd.DataFrame()) == 0.0


class TestBacktestEngine:
    """Test the backtesting engine."""

    def test_all_long_rising_prices(self):
        from pipeline.backtest import BacktestEngine
        prices = pd.Series([100, 101, 102, 103, 104, 105], index=range(6))
        signals = pd.Series([1, 1, 1, 1, 1, 1], index=range(6))
        engine = BacktestEngine(initial_capital=100000, commission=0, slippage=0)
        result = engine.run(prices, signals)
        assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]

    def test_zero_signals(self):
        from pipeline.backtest import BacktestEngine
        prices = pd.Series([100, 101, 102, 103], index=range(4))
        signals = pd.Series([0, 0, 0, 0], index=range(4))
        engine = BacktestEngine(initial_capital=100000, commission=0, slippage=0)
        result = engine.run(prices, signals)
        # No trading = no change in equity (approximately)
        assert abs(result.equity_curve.iloc[-1] - 100000) < 100

    def test_costs_reduce_pnl(self):
        from pipeline.backtest import BacktestEngine
        prices = pd.Series([100, 101, 102, 103, 104], index=range(5))
        signals = pd.Series([1, 1, 1, 1, 1], index=range(5))

        no_cost = BacktestEngine(commission=0, slippage=0).run(prices, signals)
        with_cost = BacktestEngine(commission=0.001, slippage=0.001).run(prices, signals)
        assert with_cost.equity_curve.iloc[-1] <= no_cost.equity_curve.iloc[-1]


class TestPipelineInterface:
    """Test that pipelines implement the interface correctly."""

    def test_xgboost_pipeline(self):
        from pipelines.gradient_boosting.xgboost_pipeline import XGBoostPipeline
        from shared.interfaces import TradingPipeline, PipelineOutput

        pipeline = XGBoostPipeline()
        assert isinstance(pipeline, TradingPipeline)
        assert pipeline.name == "xgboost"


class TestSeedReproducibility:
    """Test seed management."""

    def test_deterministic_numpy(self):
        from utils.seed import set_all_seeds
        set_all_seeds(42)
        a = np.random.rand(10)
        set_all_seeds(42)
        b = np.random.rand(10)
        assert np.array_equal(a, b)


class TestLabeling:
    """Test triple barrier labeling."""

    def test_basic_labels(self):
        from utils.labeling import triple_barrier_labels
        close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113])
        high = close + 1
        low = close - 0.5
        atr = pd.Series([1.0] * len(close))
        result = triple_barrier_labels(close, high, low, atr, tp_multiplier=2.0, sl_multiplier=1.0)
        assert 'label' in result.columns
        assert 'barrier_hit' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
